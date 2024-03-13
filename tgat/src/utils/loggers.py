import logging
import time
def set_logger(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    
    logger.setLevel(logging.DEBUG)
    
    # make logging file name as current date and time in DDMMYY-HHMMSS format
    logging_file_name = time.strftime("%d%m%y-%H%M%S")
    
    fh = logging.FileHandler('log/{}.log'.format(logging_file_name))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(f'%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    return logger