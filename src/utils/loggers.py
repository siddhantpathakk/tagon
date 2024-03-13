import logging
import time

def set_up_logger():
    logging_format =  f"%(levelname)s:\t%(module)s:\t%(message)s"
    
    logging.basicConfig(level=logging.INFO, format=logging_format)
    
    logger = logging.getLogger()
    
    fh = logging.FileHandler(f'./log/{time.strftime("%d%m%y-%H%M%S")}.log')

    formatter = logging.Formatter(logging_format)
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)

    return logger
