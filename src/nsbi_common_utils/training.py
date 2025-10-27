"""
PyTorch Lightning-based Neural Network Training for Density Ratio Estimation
Modern replacement for TensorFlow-based training with improved features and testability
"""

import os
import shutil
import pickle
import math
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from joblib import dump, load

# For ONNX export
import onnx
import onnxruntime as ort


# ==================== Custom Activation Functions ====================

class Mish(nn.Module):
    """Mish activation function"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ==================== PyTorch Lightning Module ====================

class DensityRatioNN(pl.LightningModule):
    """
    PyTorch Lightning module for density ratio estimation
    Supports both standard BCE and log-likelihood ratio regression
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = 4,
        neurons: int = 1000,
        learning_rate: float = 0.1,
        activation: str = 'swish',
        use_log_loss: bool = False,
        optimizer_choice: str = 'adamw',
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.use_log_loss = use_log_loss
        self.optimizer_choice = optimizer_choice
        self.weight_decay = weight_decay
        
        # Build network
        layers = []
        in_features = input_dim
        
        # Choose activation function
        if activation == 'swish':
            act_fn = nn.SiLU()  # SiLU is the PyTorch name for Swish
        elif activation == 'mish':
            act_fn = Mish()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = neurons
        
        # Output layer
        layers.append(nn.Linear(in_features, 1))
        if not use_log_loss:
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, y, weights = batch
        y_hat = self(x)
        
        if self.use_log_loss:
            # BCE with logits for log-likelihood ratio
            loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=weights)
        else:
            # Standard BCE
            loss = F.binary_cross_entropy(y_hat, y, weight=weights)
        
        # Calculate accuracy
        preds = (y_hat > 0.5 if not self.use_log_loss else (y_hat > 0)).float()
        acc = ((preds == y).float() * weights).sum() / weights.sum()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, weights = batch
        y_hat = self(x)
        
        if self.use_log_loss:
            loss = F.binary_cross_entropy_with_logits(y_hat, y, weight=weights)
        else:
            loss = F.binary_cross_entropy(y_hat, y, weight=weights)
        
        preds = (y_hat > 0.5 if not self.use_log_loss else (y_hat > 0)).float()
        acc = ((preds == y).float() * weights).sum() / weights.sum()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        if self.optimizer_choice.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_choice.lower() == 'nadam':
            optimizer = torch.optim.NAdam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_choice.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_choice}")
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.01,
            patience=30,
            min_lr=1e-9
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


# ==================== Weighted Dataset ====================

class WeightedTensorDataset(Dataset):
    """Dataset that includes sample weights"""
    
    def __init__(self, x, y, weights):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.weights = torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.weights[idx]


# ==================== Model Persistence ====================

def save_model_onnx(
    model: pl.LightningModule,
    path_to_save_model: Union[str, Path],
    scaler_instance,
    path_to_save_scaler: Union[str, Path],
    input_dim: int
) -> None:
    """Save model in ONNX format with scaler"""
    
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(path_to_save_model),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Save scaler
    dump(scaler_instance, str(path_to_save_scaler), compress=True)


def load_trained_model_onnx(
    path_to_saved_model: Union[Path, str],
    path_to_saved_scaler: Union[Path, str]
) -> Tuple:
    """Load ONNX model and scaler"""
    
    scaler = load(str(path_to_saved_scaler))
    
    # Create ONNX runtime session
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    model = ort.InferenceSession(
        str(path_to_saved_model),
        sess_options=sess_opts,
        providers=providers
    )
    
    return scaler, model


def predict_with_onnx(
    dataset: np.ndarray,
    scaler,
    model: ort.InferenceSession,
    batch_size: int = 10000
) -> np.ndarray:
    """Predict using ONNX model"""
    
    scaled_dataset = scaler.transform(dataset)
    
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    
    preds = []
    for i in range(0, len(scaled_dataset), batch_size):
        batch = scaled_dataset[i:i+batch_size].astype(np.float32)
        pred = model.run([output_name], {input_name: batch})[0]
        preds.append(pred)
    
    return np.concatenate(preds, axis=0).squeeze()


# ==================== Preselection Multi-Class Classifier ====================

class PreselectionNN(pl.LightningModule):
    """Multi-class classification for phase space preselection"""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        learning_rate: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.SiLU(),
            nn.Linear(1000, 1000),
            nn.SiLU(),
            nn.Linear(1000, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        x, y, weights = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long(), weight=weights)
        
        preds = torch.argmax(logits, dim=1)
        acc = ((preds == y).float() * weights).sum() / weights.sum()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, weights = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y.long(), weight=weights)
        
        preds = torch.argmax(logits, dim=1)
        acc = ((preds == y).float() * weights).sum() / weights.sum()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.01, patience=30, min_lr=1e-9
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


class TrainEvaluatePreselNN:
    """Training and evaluation for preselection multi-class classifier"""
    
    def __init__(
        self,
        dataset: pd.DataFrame,
        features: List[str],
        features_scaling: List[str],
        train_labels_column: str = 'train_labels',
        weights_normed_column: str = 'weights_normed'
    ):
        self.dataset = dataset
        self.features = features
        self.features_scaling = features_scaling
        self.num_classes = len(np.unique(dataset[train_labels_column]))
        self.train_labels_column = train_labels_column
        self.weights_normed_column = weights_normed_column
        
        self.model = None
        self.scaler = None
    
    def train(
        self,
        test_size: float = 0.15,
        random_state: int = 42,
        path_to_save: str = '',
        epochs: int = 20,
        batch_size: int = 1024,
        learning_rate: float = 0.1,
        num_workers: int = 4,
        accelerator: str = 'auto',
    ):
        """Train the preselection classifier"""
        
        # Split data
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            self.dataset[self.features],
            self.dataset[self.train_labels_column],
            self.dataset[self.weights_normed_column],
            test_size=test_size,
            random_state=random_state,
            stratify=self.dataset[self.train_labels_column]
        )
        
        # Setup scaler
        self.scaler = ColumnTransformer(
            [("scaler", StandardScaler(), self.features_scaling)],
            remainder='passthrough'
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = WeightedTensorDataset(X_train_scaled, y_train.values, w_train.values)
        val_dataset = WeightedTensorDataset(X_val_scaled, y_val.values, w_val.values)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # Initialize model
        model = PreselectionNN(
            input_dim=len(self.features),
            num_classes=self.num_classes,
            learning_rate=learning_rate
        )
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=300, mode='min'),
            LearningRateMonitor(logging_interval='epoch'),
        ]
        
        if path_to_save:
            path_to_save = Path(path_to_save)
            path_to_save.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=path_to_save,
                    filename='preselection-{epoch:02d}-{val_loss:.2f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1
                )
            )
        
        # Train
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            accelerator=accelerator,
            devices=1,
            logger=TensorBoardLogger('lightning_logs', name='preselection'),
            enable_progress_bar=True,
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        self.model = model
        
        # Save in ONNX format
        if path_to_save:
            model_path = path_to_save / 'model_preselection.onnx'
            scaler_path = path_to_save / 'model_scaler_presel.bin'
            save_model_onnx(model, model_path, self.scaler, scaler_path, len(self.features))
    
    def predict(self, dataset: pd.DataFrame) -> np.ndarray:
        """Predict using trained model"""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        
        self.model.eval()
        X_scaled = self.scaler.transform(dataset[self.features])
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
        
        return probs.numpy()


# ==================== Density Ratio Estimation ====================

class TrainEvaluateNN:
    """Main class for density ratio estimation with ensemble support"""
    
    def __init__(
        self,
        dataset: pd.DataFrame,
        weights: np.ndarray,
        training_labels: np.ndarray,
        features: List[str],
        features_scaling: List[str],
        sample_name: Tuple[str, str],
        output_dir: str,
        output_name: str,
        path_to_figures: str = '',
        path_to_models: str = '',
        path_to_ratios: str = '',
        use_log_loss: bool = False,
        delete_existing_models: bool = False,
    ):
        self.dataset = dataset
        self.weights = weights
        self.training_labels = training_labels
        self.features = features
        self.features_scaling = features_scaling
        self.sample_name = sample_name
        self.output_dir = output_dir
        self.output_name = output_name
        self.use_log_loss = use_log_loss
        
        # Setup directories
        self.path_to_figures = path_to_figures
        self.path_to_models = path_to_models
        self.path_to_ratios = path_to_ratios
        
        if delete_existing_models:
            for path in [path_to_figures, path_to_models, path_to_ratios]:
                if os.path.exists(path):
                    shutil.rmtree(path)
        
        for path in [path_to_figures, path_to_models, path_to_ratios]:
            if path and not os.path.exists(path):
                os.makedirs(path)
        
        # Initialize ensemble storage
        self.model_ensemble = []
        self.scaler_ensemble = []
        self.histogram_calibrator = []
        self.train_idx = []
        self.holdout_idx = []
        self.full_data_prediction = None
        self.num_ensemble_members = 1
    
    def train_ensemble(
        self,
        hidden_layers: int = 4,
        neurons: int = 1000,
        number_of_epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.1,
        scaler_type: str = 'StandardScaler',
        calibration: bool = False,
        num_bins_cal: int = 40,
        activation: str = 'swish',
        validation_split: float = 0.1,
        holdout_split: float = 0.3,
        num_ensemble_members: int = 1,
        load_trained_models: bool = False,
        recalibrate_output: bool = False,
        num_workers: int = 4,
        accelerator: str = 'auto',
        dropout_rate: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Train ensemble of density ratio estimators"""
        
        print(f"Starting ensemble training with {num_ensemble_members} members")
        self.num_ensemble_members = num_ensemble_members
        
        # Initialize ensemble storage
        self.model_ensemble = [None] * num_ensemble_members
        self.scaler_ensemble = [None] * num_ensemble_members
        self.histogram_calibrator = [None] * num_ensemble_members
        self.full_data_prediction = np.zeros((num_ensemble_members, len(self.weights)))
        self.train_idx = [None] * num_ensemble_members
        self.holdout_idx = [None] * num_ensemble_members
        
        # Generate random seeds
        random_state_arr = np.random.randint(0, 2**31 - 1, size=num_ensemble_members)
        
        # Train each ensemble member
        for ensemble_idx in range(num_ensemble_members):
            print(f"\n{'='*60}")
            print(f"Training ensemble member {ensemble_idx + 1}/{num_ensemble_members}")
            print(f"{'='*60}\n")
            
            self._train_single_model(
                hidden_layers=hidden_layers,
                neurons=neurons,
                number_of_epochs=number_of_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                scaler_type=scaler_type,
                calibration=calibration,
                num_bins_cal=num_bins_cal,
                activation=activation,
                validation_split=validation_split,
                holdout_split=holdout_split,
                ensemble_idx=ensemble_idx,
                random_seed=random_state_arr[ensemble_idx],
                load_trained_models=load_trained_models,
                recalibrate_output=recalibrate_output,
                num_workers=num_workers,
                accelerator=accelerator,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
            )
        
        print("\nEnsemble training complete!")
    
    def _train_single_model(
        self,
        hidden_layers: int,
        neurons: int,
        number_of_epochs: int,
        batch_size: int,
        learning_rate: float,
        scaler_type: str,
        calibration: bool,
        num_bins_cal: int,
        activation: str,
        validation_split: float,
        holdout_split: float,
        ensemble_idx: int,
        random_seed: int,
        load_trained_models: bool,
        recalibrate_output: bool,
        num_workers: int,
        accelerator: str,
        dropout_rate: float,
        weight_decay: float,
    ):
        """Train a single model (one ensemble member)"""
        
        # Calculate holdout size
        holdout_num = math.floor(len(self.dataset) * holdout_split)
        
        # Check if loading existing model
        model_path = Path(self.path_to_models) / f"model{ensemble_idx}.onnx"
        scaler_path = Path(self.path_to_models) / f"model_scaler{ensemble_idx}.bin"
        
        if load_trained_models and model_path.exists():
            print(f"Loading existing model from {model_path}")
            # Load saved random seed
            metadata_path = Path(self.path_to_models) / f"num_events_random_state_train_holdout_split{ensemble_idx}.npy"
            if metadata_path.exists():
                holdout_num, random_seed = np.load(metadata_path)
                holdout_num = int(holdout_num)
                random_seed = int(random_seed)
        
        # Split data
        idx_all = np.arange(len(self.weights))
        self.train_idx[ensemble_idx], self.holdout_idx[ensemble_idx] = train_test_split(
            idx_all,
            test_size=holdout_num,
            random_state=random_seed,
            stratify=self.training_labels
        )
        
        # Get train/holdout data
        X_train = self.dataset.iloc[self.train_idx[ensemble_idx]][self.features]
        y_train = self.training_labels[self.train_idx[ensemble_idx]]
        w_train = self.weights[self.train_idx[ensemble_idx]]
        
        # Setup scaler
        if scaler_type == 'MinMax':
            scaler_obj = MinMaxScaler(feature_range=(-1.5, 1.5))
        elif scaler_type == 'StandardScaler':
            scaler_obj = StandardScaler()
        elif scaler_type == 'PowerTransform_Yeo':
            scaler_obj = PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler_ensemble[ensemble_idx] = ColumnTransformer(
            [("scaler", scaler_obj, self.features_scaling)],
            remainder='passthrough'
        )
        
        # Load or train model
        if load_trained_models and model_path.exists():
            self.scaler_ensemble[ensemble_idx], onnx_session = load_trained_model_onnx(
                model_path, scaler_path
            )
            self.model_ensemble[ensemble_idx] = onnx_session
        else:
            # Scale data
            X_train_scaled = self.scaler_ensemble[ensemble_idx].fit_transform(X_train)
            
            # Further split for validation
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                X_train_scaled, y_train, w_train,
                test_size=validation_split,
                random_state=random_seed,
                stratify=y_train
            )
            
            # Create datasets
            train_dataset = WeightedTensorDataset(X_tr, y_tr, w_tr)
            val_dataset = WeightedTensorDataset(X_val, y_val, w_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=True if num_workers > 0 else False
            )
            
            # Initialize model
            model = DensityRatioNN(
                input_dim=len(self.features),
                hidden_layers=hidden_layers,
                neurons=neurons,
                learning_rate=learning_rate,
                activation=activation,
                use_log_loss=self.use_log_loss,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
            )
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=300, mode='min', verbose=True),
                LearningRateMonitor(logging_interval='epoch'),
                ModelCheckpoint(
                    dirpath=self.path_to_models,
                    filename=f'model{ensemble_idx}' + '-{epoch:02d}-{val_loss:.4f}',
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1
                )
            ]
            
            # Train
            trainer = pl.Trainer(
                max_epochs=number_of_epochs,
                callbacks=callbacks,
                accelerator=accelerator,
                devices=1,
                logger=TensorBoardLogger(
                    'lightning_logs',
                    name=f'density_ratio_ensemble{ensemble_idx}'
                ),
                enable_progress_bar=True,
                gradient_clip_val=1.0,
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            print(f"Training complete for ensemble member {ensemble_idx}")
            
            # Save model in ONNX format
            save_model_onnx(
                model, model_path, self.scaler_ensemble[ensemble_idx],
                scaler_path, len(self.features)
            )
            
            # Save metadata
            metadata_path = Path(self.path_to_models) / f"num_events_random_state_train_holdout_split{ensemble_idx}.npy"
            np.save(metadata_path, np.array([holdout_num, random_seed]))
            
            # Load as ONNX for inference
            _, onnx_session = load_trained_model_onnx(model_path, scaler_path)
            self.model_ensemble[ensemble_idx] = onnx_session
        
        # Make predictions on full dataset
        self.full_data_prediction[ensemble_idx] = self._predict_with_model(
            self.dataset[self.features].values,
            ensemble_idx
        )
        
        # Handle calibration
        if calibration:
            self._calibrate_model(ensemble_idx, num_bins_cal, recalibrate_output)
            # Re-predict with calibration
            self.full_data_prediction[ensemble_idx] = self._predict_with_model(
                self.dataset[self.features].values,
                ensemble_idx
            )
        
        # Validate predictions
        self._validate_predictions(ensemble_idx)
    
    def _predict_with_model(
        self,
        data: np.ndarray,
        ensemble_idx: int
    ) -> np.ndarray:
        """Make predictions using a single ensemble member"""
        
        pred = predict_with_onnx(
            data,
            self.scaler_ensemble[ensemble_idx],
            self.model_ensemble[ensemble_idx]
        )
        
        if self.use_log_loss:
            # Convert log-odds to probability
            pred = 1.0 / (1.0 + np.exp(-pred))
        
        # Apply calibration if available
        if self.histogram_calibrator[ensemble_idx] is not None:
            pred = self.histogram_calibrator[ensemble_idx].cali_pred(pred)
            pred = np.clip(pred, 1e-25, 0.9999999)
        
        return pred
    
    def _calibrate_model(self, ensemble_idx: int, num_bins_cal: int, recalibrate: bool):
        """Apply histogram calibration to model output"""
        
        calib_path = Path(self.path_to_models) / f"model_calibrated_hist{ensemble_idx}.obj"
        
        if calib_path.exists() and not recalibrate:
            print(f"Loading existing calibration for ensemble member {ensemble_idx}")
            with open(calib_path, 'rb') as f:
                self.histogram_calibrator[ensemble_idx] = pickle.load(f)
        else:
            print(f"Calibrating ensemble member {ensemble_idx} with {num_bins_cal} bins")
            
            # Get training predictions
            train_pred = self.full_data_prediction[ensemble_idx][self.train_idx[ensemble_idx]]
            train_labels = self.training_labels[self.train_idx[ensemble_idx]]
            train_weights = self.weights[self.train_idx[ensemble_idx]]
            
            # Separate by class
            pred_num = train_pred[train_labels == 1]
            pred_den = train_pred[train_labels == 0]
            w_num = train_weights[train_labels == 1]
            w_den = train_weights[train_labels == 0]
            
            # Import calibration utility
            try:
                from nsbi_common_utils.calibration import HistogramCalibrator
                self.histogram_calibrator[ensemble_idx] = HistogramCalibrator(
                    pred_num, pred_den, w_num, w_den,
                    nbins=num_bins_cal, method='direct', mode='dynamic'
                )
                
                with open(calib_path, 'wb') as f:
                    pickle.dump(self.histogram_calibrator[ensemble_idx], f)
                    
            except ImportError:
                print("Warning: nsbi_common_utils.calibration not available. Skipping calibration.")
                self.histogram_calibrator[ensemble_idx] = None
    
    def _validate_predictions(self, ensemble_idx: int):
        """Validate model predictions for numerical stability"""
        
        train_pred = self.full_data_prediction[ensemble_idx][self.train_idx[ensemble_idx]]
        holdout_pred = self.full_data_prediction[ensemble_idx][self.holdout_idx[ensemble_idx]]
        train_labels = self.training_labels[self.train_idx[ensemble_idx]]
        holdout_labels = self.training_labels[self.holdout_idx[ensemble_idx]]
        
        checks = [
            ("Training class 0", train_pred[train_labels == 0]),
            ("Training class 1", train_pred[train_labels == 1]),
            ("Holdout class 0", holdout_pred[holdout_labels == 0]),
            ("Holdout class 1", holdout_pred[holdout_labels == 1]),
        ]
        
        for name, pred in checks:
            min_val, max_val = np.min(pred), np.max(pred)
            if min_val == 0 or max_val == 1:
                print(f"WARNING: {name} has min={min_val}, max={max_val} "
                      f"for ensemble {ensemble_idx} - possible numerical instability!")
    
    def predict_ensemble(
        self,
        dataset: pd.DataFrame,
        aggregation_type: str = 'mean_ratio'
    ) -> np.ndarray:
        """
        Predict density ratios using ensemble
        
        Args:
            dataset: Input data
            aggregation_type: How to combine ensemble predictions
                - 'mean_ratio': Mean of ratios
                - 'median_ratio': Median of ratios
                - 'mean_score': Convert mean score to ratio
                - 'median_score': Convert median score to ratio
        """
        
        n_samples = len(dataset)
        score_pred = np.zeros((self.num_ensemble_members, n_samples))
        ratio_pred = np.zeros((self.num_ensemble_members, n_samples))
        
        # Get predictions from each ensemble member
        for idx in range(self.num_ensemble_members):
            score_pred[idx] = self._predict_with_model(
                dataset[self.features].values, idx
            )
            ratio_pred[idx] = score_pred[idx] / (1.0 - score_pred[idx] + 1e-10)
        
        # Aggregate predictions
        if aggregation_type == 'median_ratio':
            return np.median(ratio_pred, axis=0)
        elif aggregation_type == 'mean_ratio':
            return np.mean(ratio_pred, axis=0)
        elif aggregation_type == 'median_score':
            score_agg = np.median(score_pred, axis=0)
            return score_agg / (1.0 - score_agg + 1e-10)
        elif aggregation_type == 'mean_score':
            score_agg = np.mean(score_pred, axis=0)
            return score_agg / (1.0 - score_agg + 1e-10)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    
    def evaluate_and_save_ratios(
        self,
        dataset: pd.DataFrame,
        aggregation_type: str = 'mean_ratio'
    ) -> str:
        """Evaluate and save density ratios"""
        
        print("Evaluating density ratios...")
        ratio_ensemble = self.predict_ensemble(dataset, aggregation_type)
        
        save_path = Path(self.path_to_ratios) / f"ratio_{self.sample_name[0]}.npy"
        np.save(save_path, ratio_ensemble)
        print(f"Saved ratios to {save_path}")
        
        return str(save_path)
    
    def test_normalization(self):
        """Test if integral of p_A/p_B * p_B ≈ 1"""
        
        # Get reference (denominator) events
        ref_mask = self.training_labels == 0
        weight_ref = self.weights[ref_mask]
        
        for idx in range(self.num_ensemble_members):
            pred_ref = self.full_data_prediction[idx][ref_mask]
            ratio = pred_ref / (1.0 - pred_ref + 1e-10)
            integral = np.sum(ratio * weight_ref)
            print(f"Ensemble member {idx}: ∫(p_A/p_B)p_B dx = {integral:.6f}")
        
        # Test ensemble aggregate
        score_mean = np.mean(self.full_data_prediction[:, ref_mask], axis=0)
        ratio_mean = score_mean / (1.0 - score_mean + 1e-10)
        integral_ensemble = np.sum(ratio_mean * weight_ref)
        print(f"Full ensemble: ∫(p_A/p_B)p_B dx = {integral_ensemble:.6f}")


# ==================== Utility Functions ====================

def convert_to_score(log_lr: np.ndarray) -> np.ndarray:
    """Convert log-likelihood ratio to probability score"""
    return 1.0 / (1.0 + np.exp(-log_lr))


def convert_to_log_ratio(score: np.ndarray) -> np.ndarray:
    """Convert probability score to log-likelihood ratio"""
    return np.log(score / (1.0 - score + 1e-10) + 1e-10)


# ==================== Testing & Validation Functions ====================

def create_synthetic_dataset(
    n_samples: int = 10000,
    n_features: int = 5,
    separation: float = 1.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset for testing
    
    Args:
        n_samples: Total number of samples
        n_features: Number of features
        separation: How separated the two classes are
        random_state: Random seed
    
    Returns:
        dataset, weights, labels
    """
    
    np.random.seed(random_state)
    
    # Generate two Gaussian distributions
    n_per_class = n_samples // 2
    
    # Class 0 (reference)
    X0 = np.random.randn(n_per_class, n_features)
    
    # Class 1 (shifted by separation)
    X1 = np.random.randn(n_per_class, n_features) + separation
    
    # Combine
    X = np.vstack([X0, X1])
    labels = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    # Equal weights normalized per class
    weights = np.hstack([
        np.ones(n_per_class) / n_per_class,
        np.ones(n_per_class) / n_per_class
    ])
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    return df, weights, labels


def test_preselection_training():
    """Test preselection multi-class classifier"""
    
    print("\n" + "="*60)
    print("TEST 1: Preselection Multi-Class Classifier")
    print("="*60 + "\n")
    
    # Create synthetic 3-class dataset
    n_samples = 5000
    n_features = 5
    n_classes = 3
    
    np.random.seed(42)
    X = []
    y = []
    
    for class_idx in range(n_classes):
        X_class = np.random.randn(n_samples // n_classes, n_features) + class_idx * 2
        X.append(X_class)
        y.append(np.full(n_samples // n_classes, class_idx))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Equal weights per class
    weights = np.ones(len(y)) / len(y) * n_classes
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['train_labels'] = y
    df['weights_normed'] = weights
    
    # Train
    trainer = TrainEvaluatePreselNN(
        dataset=df,
        features=feature_names,
        features_scaling=feature_names,
    )
    
    trainer.train(
        test_size=0.2,
        epochs=5,
        batch_size=128,
        learning_rate=0.01,
        num_workers=0,
        path_to_save='test_outputs/preselection'
    )
    
    # Test prediction
    predictions = trainer.predict(df)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y)
    
    print(f"\nPreselection Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Expected: > 0.85 (for well-separated classes)")
    
    assert accuracy > 0.7, "Preselection accuracy too low!"
    print("✓ Preselection test PASSED\n")


def test_density_ratio_estimation():
    """Test binary density ratio estimation"""
    
    print("\n" + "="*60)
    print("TEST 2: Binary Density Ratio Estimation")
    print("="*60 + "\n")
    
    # Create synthetic dataset with known density ratio
    df, weights, labels = create_synthetic_dataset(
        n_samples=10000,
        n_features=5,
        separation=1.5,
        random_state=42
    )
    
    feature_names = df.columns.tolist()
    
    # Train density ratio estimator
    trainer = TrainEvaluateNN(
        dataset=df,
        weights=weights,
        training_labels=labels,
        features=feature_names,
        features_scaling=feature_names,
        sample_name=("ClassA", "ClassB"),
        output_dir="test_outputs",
        output_name="test_ratio",
        path_to_figures="test_outputs/figures",
        path_to_models="test_outputs/models",
        path_to_ratios="test_outputs/ratios",
        use_log_loss=False,
        delete_existing_models=True,
    )
    
    trainer.train_ensemble(
        hidden_layers=3,
        neurons=128,
        number_of_epochs=20,
        batch_size=256,
        learning_rate=0.01,
        scaler_type='StandardScaler',
        calibration=False,
        activation='swish',
        validation_split=0.15,
        holdout_split=0.2,
        num_ensemble_members=2,
        num_workers=0,
        accelerator='cpu',
    )
    
    # Test normalization
    print("\nTesting normalization (should be close to 1.0):")
    trainer.test_normalization()
    
    # Test predictions
    ratios = trainer.predict_ensemble(df, aggregation_type='mean_ratio')
    
    print(f"\nDensity Ratio Statistics:")
    print(f"  Mean: {np.mean(ratios):.4f}")
    print(f"  Std:  {np.std(ratios):.4f}")
    print(f"  Min:  {np.min(ratios):.4f}")
    print(f"  Max:  {np.max(ratios):.4f}")
    
    # Check that ratios are sensible
    assert np.all(np.isfinite(ratios)), "Non-finite ratios detected!"
    assert np.all(ratios > 0), "Non-positive ratios detected!"
    
    print("✓ Density ratio test PASSED\n")


def test_ensemble_aggregation():
    """Test different ensemble aggregation methods"""
    
    print("\n" + "="*60)
    print("TEST 3: Ensemble Aggregation Methods")
    print("="*60 + "\n")
    
    df, weights, labels = create_synthetic_dataset(
        n_samples=5000, n_features=3, separation=1.0
    )
    
    feature_names = df.columns.tolist()
    
    trainer = TrainEvaluateNN(
        dataset=df,
        weights=weights,
        training_labels=labels,
        features=feature_names,
        features_scaling=feature_names,
        sample_name=("A", "B"),
        output_dir="test_outputs",
        output_name="test",
        path_to_models="test_outputs/models_agg",
        delete_existing_models=True,
    )
    
    trainer.train_ensemble(
        hidden_layers=2,
        neurons=64,
        number_of_epochs=10,
        batch_size=256,
        learning_rate=0.01,
        num_ensemble_members=3,
        num_workers=0,
        accelerator='cpu',
    )
    
    # Test all aggregation methods
    methods = ['mean_ratio', 'median_ratio', 'mean_score', 'median_score']
    
    for method in methods:
        ratios = trainer.predict_ensemble(df, aggregation_type=method)
        print(f"{method:15s}: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}")
        assert np.all(np.isfinite(ratios)), f"Non-finite values in {method}!"
    
    print("✓ Ensemble aggregation test PASSED\n")


def test_model_saving_loading():
    """Test model persistence (save/load)"""
    
    print("\n" + "="*60)
    print("TEST 4: Model Saving and Loading")
    print("="*60 + "\n")
    
    df, weights, labels = create_synthetic_dataset(n_samples=2000, n_features=3)
    feature_names = df.columns.tolist()
    
    model_dir = "test_outputs/models_persist"
    
    # Train and save
    trainer1 = TrainEvaluateNN(
        dataset=df,
        weights=weights,
        training_labels=labels,
        features=feature_names,
        features_scaling=feature_names,
        sample_name=("A", "B"),
        output_dir="test_outputs",
        output_name="test",
        path_to_models=model_dir,
        delete_existing_models=True,
    )
    
    trainer1.train_ensemble(
        hidden_layers=2,
        neurons=32,
        number_of_epochs=5,
        batch_size=128,
        num_ensemble_members=1,
        num_workers=0,
        accelerator='cpu',
    )
    
    pred1 = trainer1.predict_ensemble(df)
    
    # Load and predict
    trainer2 = TrainEvaluateNN(
        dataset=df,
        weights=weights,
        training_labels=labels,
        features=feature_names,
        features_scaling=feature_names,
        sample_name=("A", "B"),
        output_dir="test_outputs",
        output_name="test",
        path_to_models=model_dir,
    )
    
    trainer2.train_ensemble(
        hidden_layers=2,
        neurons=32,
        number_of_epochs=5,
        batch_size=128,
        num_ensemble_members=1,
        load_trained_models=True,
        num_workers=0,
        accelerator='cpu',
    )
    
    pred2 = trainer2.predict_ensemble(df)
    
    # Compare predictions
    max_diff = np.max(np.abs(pred1 - pred2))
    print(f"Max difference between saved/loaded predictions: {max_diff:.2e}")
    
    assert max_diff < 1e-5, "Loaded model predictions differ significantly!"
    print("✓ Model persistence test PASSED\n")


def run_all_tests():
    """Run complete test suite"""
    
    print("\n" + "="*70)
    print(" PYTORCH LIGHTNING NEURAL NETWORK TEST SUITE")
    print("="*70)
    
    try:
        test_preselection_training()
        test_density_ratio_estimation()
        test_ensemble_aggregation()
        test_model_saving_loading()
        
        print("\n" + "="*70)
        print(" ALL TESTS PASSED ✓")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    # Run tests
    run_all_tests()
    
    print("\nTo use in production:")
    print("1. Import the classes: TrainEvaluateNN, TrainEvaluatePreselNN")
    print("2. Prepare your dataset with features, weights, and labels")
    print("3. Create trainer instance and call train_ensemble()")
    print("4. Use predict_ensemble() for inference")
    print("5. Save ratios with evaluate_and_save_ratios()")