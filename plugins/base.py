import os
import importlib
from typing import Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class PluginAction:
    name: str
    description: str
    params: dict = field(default_factory=dict)
    perform: Optional[Callable] = None


class BasePlugin:
    def __init__(self, agent: Any = None):
        self.agent = agent
        self.name = self.__class__.__name__

    def init(self):
        pass

    def get_actions(self) -> list[PluginAction]:
        return []


class PluginManager:
    def __init__(self, plugin_dir: str = "./plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: dict[str, BasePlugin] = {}
        self.actions: dict[str, PluginAction] = {}

    def discover_plugins(self) -> list[str]:
        if not os.path.exists(self.plugin_dir):
            return []

        plugins = []
        for item in os.listdir(self.plugin_dir):
            plugin_path = os.path.join(self.plugin_dir, item)
            if os.path.isdir(plugin_path):
                main_file = os.path.join(plugin_path, "main.py")
                if os.path.exists(main_file):
                    plugins.append(item)

        return plugins

    async def load_plugin(self, name: str, agent: Any = None) -> Optional[BasePlugin]:
        plugin_path = os.path.join(self.plugin_dir, name, "main.py")
        if not os.path.exists(plugin_path):
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                f"plugins.{name}.main", plugin_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "PluginInstance"):
                plugin = module.PluginInstance(agent)
                plugin.init()
                self.plugins[name] = plugin

                for action in plugin.get_actions():
                    self.actions[action.name] = action

                return plugin
        except Exception as e:
            print(f"Failed to load plugin {name}: {e}")

        return None

    async def load_all(self, agent: Any = None):
        plugin_names = self.discover_plugins()
        for name in plugin_names:
            await self.load_plugin(name, agent)

    def get_action(self, name: str) -> Optional[PluginAction]:
        return self.actions.get(name)

    def list_actions(self) -> list[PluginAction]:
        return list(self.actions.values())

    def has_action(self, name: str) -> bool:
        return name in self.actions
