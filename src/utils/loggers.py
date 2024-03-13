import logging
import time

def set_up_logger():
    logging_format =  f"%(levelname)s:\t%(module)s:\t%(message)s"
    logging.basicConfig(level=logging.INFO, format=logging_format)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(f'./log/{time.strftime("%d%m%y-%H%M%S")}.log')
    # fh.setLevel(logging.DEBUG)
    
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.WARN)
    

    formatter = logging.Formatter(logging_format)
    fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    # logger.addHandler(ch)

    return logger
