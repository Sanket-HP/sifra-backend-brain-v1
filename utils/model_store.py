# utils/model_store.py

import os
import pickle
import json
from datetime import datetime


class ModelStore:
    """
    ModelStore
    ----------
    A utility class for saving/loading ML models used by SIFRA AI.

    Features:
        ✔ Save model to disk
        ✔ Load model from disk
        ✔ Save model metadata (JSON)
        ✔ Auto versioning (model_v1, model_v2, ...)
        ✔ Validation checks
        ✔ Production-ready serialization
    """

    def __init__(self, base_dir="saved_models"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Find next available version
    # -------------------------------------------------------------
    def _get_next_version(self, model_name):
        version = 1
        while os.path.exists(f"{self.base_dir}/{model_name}_v{version}.pkl"):
            version += 1
        return version

    # -------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------
    def save(self, model, model_name, metadata=None):
        """
        Save ML model with versioning and optional metadata.
        """
        version = self._get_next_version(model_name)

        model_path = f"{self.base_dir}/{model_name}_v{version}.pkl"
        meta_path = f"{self.base_dir}/{model_name}_v{version}.json"

        # Save model file
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Write metadata
        meta = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": model_path,
        }

        if metadata:
            meta.update(metadata)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        return {
            "status": "success",
            "message": f"Model saved as version {version}",
            "model_path": model_path,
            "metadata_path": meta_path,
            "version": version
        }

    # -------------------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------------------
    def load(self, model_name, version=None):
        """
        Load model by name and version.
        If version=None, load latest.
        """
        # Load latest version
        if version is None:
            version = self._get_latest_version(model_name)
            if version is None:
                return {"error": "Model not found."}

        model_path = f"{self.base_dir}/{model_name}_v{version}.pkl"
        meta_path = f"{self.base_dir}/{model_name}_v{version}.json"

        if not os.path.exists(model_path):
            return {"error": f"Model version {version} not found."}

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        metadata = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return {
            "status": "success",
            "model": model,
            "metadata": metadata
        }

    # -------------------------------------------------------------
    # GET LATEST VERSION
    # -------------------------------------------------------------
    def _get_latest_version(self, model_name):
        """
        Detect the highest saved version number.
        """
        versions = []
        for file in os.listdir(self.base_dir):
            if file.startswith(model_name) and file.endswith(".pkl"):
                try:
                    v = int(file.replace(model_name + "_v", "").replace(".pkl", ""))
                    versions.append(v)
                except:
                    pass
        return max(versions) if versions else None

    # -------------------------------------------------------------
    # LIST ALL MODELS
    # -------------------------------------------------------------
    def list_models(self):
        """
        List all saved models and versions.
        """
        models = {}

        for file in os.listdir(self.base_dir):
            if file.endswith(".pkl"):
                name, version = file.replace(".pkl", "").split("_v")
                version = int(version)
                if name not in models:
                    models[name] = []
                models[name].append(version)

        return models

