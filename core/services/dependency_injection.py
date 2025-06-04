"""
Dependency Injection Container
=============================

Container d'injection de dépendances pour découpler les composants.
Gère les instances de services et repositories de manière centralisée.

Auteur: khalid
Date: 04/06/2025
"""

from typing import TypeVar, Type, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Container:
    """Container d'injection de dépendances."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._initializers: Dict[str, Callable] = {}
    
    def register_singleton(
        self, 
        interface: Type[T], 
        implementation: Type[T],
        initializer: Optional[Callable] = None
    ) -> None:
        """Enregistre un service en singleton.
        
        Args:
            interface: Interface ou classe abstraite
            implementation: Implémentation concrète
            initializer: Fonction d'initialisation optionnelle
        """
        key = self._get_key(interface)
        self._factories[key] = implementation
        
        if initializer:
            self._initializers[key] = initializer
        
        logger.debug(f"Registered singleton: {key} -> {implementation.__name__}")
    
    def register_transient(
        self, 
        interface: Type[T], 
        implementation: Type[T]
    ) -> None:
        """Enregistre un service en mode transient (nouvelle instance à chaque fois).
        
        Args:
            interface: Interface ou classe abstraite  
            implementation: Implémentation concrète
        """
        key = self._get_key(interface)
        self._factories[key] = implementation
        logger.debug(f"Registered transient: {key} -> {implementation.__name__}")
    
    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Enregistre une instance existante.
        
        Args:
            interface: Interface ou classe abstraite
            instance: Instance à enregistrer
        """
        key = self._get_key(interface)
        self._services[key] = instance
        logger.debug(f"Registered instance: {key}")
    
    def resolve(self, interface: Type[T]) -> T:
        """Résout une dépendance.
        
        Args:
            interface: Interface à résoudre
            
        Returns:
            Instance du service
            
        Raises:
            DependencyError: Si la dépendance n'est pas trouvée
        """
        key = self._get_key(interface)
        
        # Instance déjà enregistrée
        if key in self._services:
            return self._services[key]
        
        # Singleton déjà créé
        if key in self._singletons:
            return self._singletons[key]
        
        # Factory disponible
        if key in self._factories:
            factory = self._factories[key]
            
            try:
                # Résolution des dépendances du constructeur
                dependencies = self._resolve_dependencies(factory)
                instance = factory(**dependencies)
                
                # Initialisation si nécessaire
                if key in self._initializers:
                    self._initializers[key](instance)
                
                # Stockage si singleton
                if key not in self._services:  # Pas transient
                    self._singletons[key] = instance
                
                logger.debug(f"Resolved: {key}")
                return instance
                
            except Exception as e:
                logger.error(f"Error resolving {key}: {str(e)}")
                raise DependencyError(f"Cannot resolve {interface.__name__}: {str(e)}")
        
        raise DependencyError(f"No registration found for {interface.__name__}")
    
    async def resolve_async(self, interface: Type[T]) -> T:
        """Résout une dépendance de manière asynchrone."""
        return self.resolve(interface)
    
    def _resolve_dependencies(self, factory: Type) -> Dict[str, Any]:
        """Résout les dépendances du constructeur."""
        dependencies = {}
        
        # Inspection des annotations de type du constructeur
        if hasattr(factory, '__init__'):
            annotations = getattr(factory.__init__, '__annotations__', {})
            
            for param_name, param_type in annotations.items():
                if param_name == 'return':
                    continue
                
                try:
                    dependencies[param_name] = self.resolve(param_type)
                except DependencyError:
                    # Dépendance optionnelle ou primitive
                    logger.debug(f"Could not resolve dependency {param_name}: {param_type}")
        
        return dependencies
    
    def _get_key(self, interface: Type) -> str:
        """Génère une clé unique pour l'interface."""
        return f"{interface.__module__}.{interface.__name__}"
    
    def clear(self) -> None:
        """Vide le container."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._initializers.clear()
        logger.debug("Container cleared")
    
    async def dispose(self) -> None:
        """Dispose les ressources du container."""
        for service in self._services.values():
            if hasattr(service, 'dispose'):
                try:
                    if asyncio.iscoroutinefunction(service.dispose):
                        await service.dispose()
                    else:
                        service.dispose()
                except Exception as e:
                    logger.error(f"Error disposing service: {e}")
        
        for singleton in self._singletons.values():
            if hasattr(singleton, 'dispose'):
                try:
                    if asyncio.iscoroutinefunction(singleton.dispose):
                        await singleton.dispose()
                    else:
                        singleton.dispose()
                except Exception as e:
                    logger.error(f"Error disposing singleton: {e}")
        
        self.clear()
        logger.info("Container disposed")


class DependencyError(Exception):
    """Exception pour les erreurs d'injection de dépendances."""
    pass


# Instance globale du container
container = Container()


def get_container() -> Container:
    """Récupère l'instance globale du container."""
    return container


def inject(interface: Type[T]) -> T:
    """Fonction utilitaire pour l'injection de dépendances.
    
    Args:
        interface: Interface à injecter
        
    Returns:
        Instance du service
    """
    return container.resolve(interface)


async def inject_async(interface: Type[T]) -> T:
    """Fonction utilitaire pour l'injection asynchrone.
    
    Args:
        interface: Interface à injecter
        
    Returns:
        Instance du service
    """
    return await container.resolve_async(interface)
