import yaml
from loguru import logger
from pathlib import Path

def load_config(config_path="config.yaml"):

    config_path = Path(__file__).parent / "config.yaml"
    try: 
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.success("Конфиг успешно загружен")
        return config
    except FileNotFoundError:
        logger.error(f"Ошибка: файл {config_path} не найден.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML: {e}")
        return None
    except Exception as e:
        logger.error(f"Неизвестная ошибка при попытке загрузки config.yaml : {e}")
        return None
