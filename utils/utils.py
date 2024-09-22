
import logging
import sys

def setup_logging(logger_level=logging.INFO, log_file="data_preparation.log"):
    """
    Configures the logging settings.
    """
    logger = logging.basicConfig(
        level=logger_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    return logger

def convert_target_str_to_int(df):
    assert "Attrition" in df.columns
    target_class_str_mapper = {
        "No": 0,
        "Yes": 1,
    }
    target_class_str = "Attrition"
    target_class = "target"
    df[target_class] = df[target_class_str].apply(lambda x: target_class_str_mapper[x])
    
    return df