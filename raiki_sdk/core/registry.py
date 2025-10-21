# core/registry.py
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass
import importlib
from pathlib import Path
from raiki_sdk.core import _normalize_version_str, _normalize_category_str
import logging

_logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Metadata about a cached model"""
    category: str       # "rolex", "lv", "redframe"
    part: str          # "P01", "P02", etc.
    version: str       # "v1", "v2", etc.
    package_path: str  # Full import path like "raiki_sdk.model_packages.rolex.rolex_p01_v1"
    package_name: str  # "rolex_p01_v1_3", etc.
    detector_factory: Optional[Callable] = None  # Factory function
    authenticator_factory: Optional[Callable] = None  # Factory function
    classifier_factory: Optional[Callable] = None # Factory function

    # Cache instances
    _detector: Optional[Any] = None
    _authenticator: Optional[Any] = None
    _classifier: Optional[Any] = None

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
    
    def _construct_package_path(self, category: str, part: str, version: str) -> tuple[str, str]:
        """Construct the import path for a model package"""
        version = _normalize_version_str(version)

        package_name = f"{category}_{part}_{version}".lower()
        return f"raiki_sdk.model_packages.{category}.{package_name}", package_name

    def _load_classifier_package(self, category: str, part: str, version: str) -> ModelInfo:
        """Dynamically load a classifier package and cache its factory functions"""
        category = _normalize_category_str(category)
        package_path, package_name = self._construct_package_path(category, part, version)
        key = package_name
        
        try:
            # Import the classifier package
            module = importlib.import_module(package_path)

            # Get factory functions from the module
            classifier_factory = getattr(module, 'get_classifier', None)

            if not classifier_factory:
                raise AttributeError(f"Model package {package_path} must provide 'get_classifier' function")
            
            # Create and cache model info
            model_info = ModelInfo(
                category=category,
                part=part,
                version=version,
                package_path=package_path,
                package_name=package_name,
                classifier_factory=classifier_factory
            )
            
            self._models[key] = model_info
            _logger.info(f"✅ Loaded classifier package: {category} - {part} - {version}")
            
            return model_info

        except ImportError as e:
            raise ImportError(f"Failed to import classifier package {package_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Invalid classifier package {package_path}: {e}")
    
    def _load_auth_model_package(self, category: str, part: str, version: str) -> ModelInfo:
        """Dynamically load a model package and cache its factory functions"""
        package_path, package_name = self._construct_package_path(category, part, version)
        key = package_name
        
        try:
            # Import the model package
            module = importlib.import_module(package_path)
            
            # Get factory functions from the module
            detector_factory = getattr(module, 'get_detector', None)
            authenticator_factory = getattr(module, 'get_authenticator', None)
            
            if not detector_factory or not authenticator_factory:
                raise AttributeError(f"Model package {package_path} must provide 'get_detector' and 'get_authenticator' functions")
            
            # Create and cache model info
            model_info = ModelInfo(
                category=category,
                part=part,
                version=version,
                package_path=package_path,
                package_name=package_name,
                detector_factory=detector_factory,
                authenticator_factory=authenticator_factory
            )
            
            self._models[key] = model_info
            _logger.info(f"✅ Loaded model package: {category} - {part} - {version}")
            
            return model_info
            
        except ImportError as e:
            raise ImportError(f"Failed to import model package {package_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Invalid model package {package_path}: {e}")

    def get_classifier_model(self, category: str, part: str, model_version: str) -> dict:
        """
        Get classifier model package - loads models lazily on first access
        
        Returns:
            dict with 'classifier' instance
        Args:
            category: Brand category ("rolex", "lv", etc.)
            part: Only P00 is supported for classifier models
            model_version: Version identifier ("v1_1", "v1_2", etc.)
        """
        category = _normalize_category_str(category)
        _, key = self._construct_package_path(category, part, model_version)

        if part.lower() != "p00":
            raise ValueError(f"Only P00 is supported for classifier models, got {part}")

        # Load model package if not already cached
        if key not in self._models:
            model_info = self._load_classifier_package(category, part, model_version)
        else:
            model_info = self._models[key]

        if model_info._classifier is None:
            _logger.info(f"Loading classifier for {category} - {part} - {model_version}")
            model_info._classifier = model_info.classifier_factory()

        return {
            'category': model_info.category,
            'part': model_info.part,
            'version': model_info.version,
            'package_path': model_info.package_path,
            'classifier': model_info._classifier
        }

    def get_auth_model(self, category: str, part: str, model_version: str) -> dict:
        """
        Get model package - loads models lazily on first access
        
        Args:
            category: Brand category ("rolex", "lv", etc.)
            part: Part identifier ("P01", "P02", etc.) 
            model_version: Version identifier ("v1_1", "v1_2", etc.)
        
        Returns:
            dict with 'detector' and 'authenticator' instances
        """
        _, key = self._construct_package_path(category, part, model_version)

        # Load model package if not already cached
        if key not in self._models:
            model_info = self._load_auth_model_package(category, part, model_version)
        else:
            model_info = self._models[key]

        # Lazy load: Only create instances when actually needed
        if model_info._detector is None:
            _logger.info(f"Loading detector for {category} - {part} - {model_version}")
            model_info._detector = model_info.detector_factory()

        if model_info._authenticator is None:
            _logger.info(f"Loading authenticator for {category} - {part} - {model_version}")
            model_info._authenticator = model_info.authenticator_factory()

        return {
            'category': model_info.category,
            'part': model_info.part,
            'version': model_info.version,
            'package_path': model_info.package_path,
            'detector': model_info._detector,
            'authenticator': model_info._authenticator
        }

    def get_model(self, category: str, part: str, model_version: str) -> dict:
        """
        Get model package - automatically determines if it's auth or classifier model
        
        Args:
            category: Brand category ("rolex", "lv", etc.)
            part: Part identifier ("P00" for classifier, "P01", "P02", etc. for auth)
            model_version: Version identifier ("v1_1", "v1_2", etc.)
        
        Returns:
            dict with appropriate model instances based on part type
        """
        if part.lower() == "p00":
            return self.get_classifier_model(category, part, model_version)
        else:
            return self.get_auth_model(category, part, model_version)

    def list_models(self) -> dict:
        """List all cached models (without loading them)"""
        return {
            key: {
                'category': info.category,
                'part': info.part,
                'version': info.version,
                'package_path': info.package_path,
                'loaded': info._detector is not None or info._authenticator is not None or info._classifier is not None
            }
            for key, info in self._models.items()
        }
    
    def list_available_models(self) -> List[dict]:
        """
        Scan and list all available model packages without loading them
        
        Returns:
            List of available models with their metadata
        """
        available_models = []
        model_packages_path = Path(__file__).parent.parent / "model_packages"
        
        if not model_packages_path.exists():
            return available_models
        
        # Scan for brand directories
        for brand_dir in model_packages_path.iterdir():
            if brand_dir.is_dir() and not brand_dir.name.startswith('_'):
                # Scan for model packages within brand
                for model_dir in brand_dir.iterdir():
                    if model_dir.is_dir() and not model_dir.name.startswith('_'):
                        # Parse model package name (e.g., "rolex_p01_v1")
                        try:
                            parts = model_dir.name.split('_')
                            if len(parts) >= 3:
                                category = parts[0]
                                part = parts[1].upper()
                                version = '_'.join(parts[2:])  # Handle versions like "v1_1"
                                
                                available_models.append({
                                    'category': category,
                                    'part': part,
                                    'version': version,
                                    'package_name': model_dir.name,
                                    'package_path': f"raiki_sdk.model_packages.{brand_dir.name}.{model_dir.name}"
                                })
                        except Exception:
                            # Skip malformed package names
                            continue
        
        return available_models

# Global singleton
registry = ModelRegistry()