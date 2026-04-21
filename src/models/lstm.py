from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class LSTMConfig:
	sequence_length: int = 30
	batch_size: int = 256
	hidden_size: int = 128
	num_layers: int = 2
	dropout: float = 0.2
	learning_rate: float = 1e-3
	epochs: int = 7


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = np.clip(y_true, a_min=0.0, a_max=None)
	y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
	return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def load_processed_splits(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	processed = project_root / "data" / "processed"
	train_df = pd.read_parquet(processed / "train.parquet")
	val_df = pd.read_parquet(processed / "val.parquet")
	test_df = pd.read_parquet(processed / "test.parquet")

	for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
		frame["date"] = pd.to_datetime(frame["date"])
		frame["split"] = split_name

	return train_df, val_df, test_df


def preprocess_and_scale(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	target_col: str = "sales",
	group_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
	if group_cols is None:
		group_cols = ["store_nbr", "family"]

	full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
	full_df = full_df.sort_values(group_cols + ["date"]).reset_index(drop=True)

	# Keep family numeric and stable across splits.
	if "family" in full_df.columns and full_df["family"].dtype == object:
		vocab = {name: idx for idx, name in enumerate(sorted(train_df["family"].astype(str).unique()))}
		full_df["family"] = full_df["family"].astype(str).map(vocab).fillna(-1).astype(np.int16)

	exclude = {"date", target_col, "split"}
	feature_cols = [c for c in full_df.columns if c not in exclude]

	scaler = StandardScaler()
	train_mask = full_df["split"] == "train"
	scaler.fit(full_df.loc[train_mask, feature_cols])

	full_df.loc[:, feature_cols] = scaler.transform(full_df[feature_cols]).astype(np.float32)
	full_df.loc[:, target_col] = full_df[target_col].astype(np.float32)

	return full_df, feature_cols


def create_sequences(
	df: pd.DataFrame,
	feature_cols: List[str],
	sequence_length: int,
	target_split: str,
	target_col: str = "sales",
	group_cols: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Build fixed-length rolling sequences per (store_nbr, family).

	For each sample:
	- X: previous `sequence_length` rows of features
	- y: current row sales (next-day relative to X window end)

	Only rows whose target belongs to `target_split` are yielded, while history can come
	from earlier rows (including prior split), which avoids future leakage.
	"""
	if group_cols is None:
		group_cols = ["store_nbr", "family"]

	df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

	X_list: List[np.ndarray] = []
	y_list: List[float] = []

	for _, grp in df.groupby(group_cols, sort=False):
		grp = grp.reset_index(drop=True)
		feat = grp[feature_cols].to_numpy(dtype=np.float32)
		target = grp[target_col].to_numpy(dtype=np.float32)
		split = grp["split"].to_numpy()

		for i in range(sequence_length, len(grp)):
			if split[i] != target_split:
				continue
			X_list.append(feat[i - sequence_length : i])
			y_list.append(float(target[i]))

	if not X_list:
		raise ValueError(f"No sequences created for split '{target_split}'.")

	X = np.stack(X_list).astype(np.float32)
	y = np.array(y_list, dtype=np.float32)
	return X, y


class SalesSequenceDataset(Dataset):
	def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
		self.X = torch.tensor(X, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)

	def __len__(self) -> int:
		return len(self.y)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
	def __init__(
		self,
		input_size: int,
		hidden_size: int = 128,
		num_layers: int = 2,
		dropout: float = 0.2,
	) -> None:
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		self.fc = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out, _ = self.lstm(x)
		last_hidden = out[:, -1, :]
		pred = self.fc(last_hidden)
		return pred.squeeze(-1)


def build_dataloaders(
	full_df: pd.DataFrame,
	feature_cols: List[str],
	config: LSTMConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
	X_train, y_train = create_sequences(full_df, feature_cols, config.sequence_length, "train")
	X_val, y_val = create_sequences(full_df, feature_cols, config.sequence_length, "val")
	X_test, y_test = create_sequences(full_df, feature_cols, config.sequence_length, "test")

	train_ds = SalesSequenceDataset(X_train, y_train)
	val_ds = SalesSequenceDataset(X_val, y_val)
	test_ds = SalesSequenceDataset(X_test, y_test)

	train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
	val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)
	test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)

	sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
	return train_loader, val_loader, test_loader, sizes


def train_lstm(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device,
	epochs: int = 7,
	lr: float = 1e-3,
) -> None:
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	model.to(device)
	for epoch in range(1, epochs + 1):
		model.train()
		train_losses: List[float] = []

		for X_batch, y_batch in train_loader:
			X_batch = X_batch.to(device)
			y_batch = y_batch.to(device)

			optimizer.zero_grad()
			preds = model(X_batch)
			loss = criterion(preds, y_batch)
			loss.backward()
			optimizer.step()

			train_losses.append(float(loss.item()))

		model.eval()
		val_losses: List[float] = []
		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				X_batch = X_batch.to(device)
				y_batch = y_batch.to(device)
				preds = model(X_batch)
				loss = criterion(preds, y_batch)
				val_losses.append(float(loss.item()))

		print(
			f"Epoch {epoch:02d}/{epochs} | "
			f"Train MSE: {np.mean(train_losses):.5f} | "
			f"Val MSE: {np.mean(val_losses):.5f}"
		)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
	model.eval()
	y_true_all: List[np.ndarray] = []
	y_pred_all: List[np.ndarray] = []

	with torch.no_grad():
		for X_batch, y_batch in loader:
			X_batch = X_batch.to(device)
			preds = model(X_batch).cpu().numpy()
			y_pred_all.append(preds)
			y_true_all.append(y_batch.numpy())

	y_true = np.concatenate(y_true_all)
	y_pred = np.concatenate(y_pred_all)
	y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
	return y_true, y_pred


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	return {
		"MAE": float(mean_absolute_error(y_true, y_pred)),
		"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
		"RMSLE": float(rmsle(y_true, y_pred)),
	}


def compare_with_baselines(lstm_metrics: Dict[str, float], baselines: Dict[str, Dict[str, float]]) -> pd.DataFrame:
	rows = [{"model": "LSTM", **lstm_metrics}]
	for name, metrics in baselines.items():
		rows.append({"model": name, **metrics})
	result = pd.DataFrame(rows)
	return result.sort_values("RMSLE", ascending=True).reset_index(drop=True)


def run_lstm_pipeline(project_root: Path | None = None, config: LSTMConfig | None = None) -> Dict[str, Dict[str, float]]:
	if project_root is None:
		project_root = Path(__file__).resolve().parents[2]
	if config is None:
		config = LSTMConfig()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	train_df, val_df, test_df = load_processed_splits(project_root)
	full_df, feature_cols = preprocess_and_scale(train_df, val_df, test_df)
	train_loader, val_loader, test_loader, sizes = build_dataloaders(full_df, feature_cols, config)

	print("Sequence counts:", sizes)
	print("Input size:", len(feature_cols))

	model = LSTMModel(
		input_size=len(feature_cols),
		hidden_size=config.hidden_size,
		num_layers=config.num_layers,
		dropout=config.dropout,
	)

	train_lstm(
		model=model,
		train_loader=train_loader,
		val_loader=val_loader,
		device=device,
		epochs=config.epochs,
		lr=config.learning_rate,
	)

	y_val, val_pred = predict(model, val_loader, device)
	y_test, test_pred = predict(model, test_loader, device)

	val_metrics = evaluate_predictions(y_val, val_pred)
	test_metrics = evaluate_predictions(y_test, test_pred)

	print("Validation metrics:", val_metrics)
	print("Test metrics:", test_metrics)

	# Update these numbers from your baseline notebook if needed.
	baseline_metrics = {
		"Naive": {"MAE": np.nan, "RMSE": np.nan, "RMSLE": np.nan},
		"Ridge": {"MAE": np.nan, "RMSE": np.nan, "RMSLE": np.nan},
		"RandomForest": {"MAE": np.nan, "RMSE": np.nan, "RMSLE": 1.20},
	}
	comparison = compare_with_baselines(val_metrics, baseline_metrics)
	print("\nModel comparison (lower is better):")
	print(comparison)

	return {"val": val_metrics, "test": test_metrics}


if __name__ == "__main__":
	run_lstm_pipeline()
