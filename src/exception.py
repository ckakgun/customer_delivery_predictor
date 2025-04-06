import sys
from src.logger import logging

def error_msg_detail(error, error_detail:sys):
    """Format error message with file name, line number, and error details."""
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """Custom exception class for handling and formatting error messages with detailed information."""
    def __init__(self, error_message, error_detail:sys):
        """Initialize CustomException with error message and system error details."""
        super().__init__(error_message)
        self.error_message = error_msg_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error message when exception is created

    def __str__(self):
        return self.error_message
    
if __name__=='__main__':
    try:
        a = 1/0
    except Exception as e: 
        logging.info("Divide by zero")
        raise CustomException(e, sys) from e
