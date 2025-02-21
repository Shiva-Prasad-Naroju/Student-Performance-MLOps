import sys # Importing sys module to work with system-specific parameters and functions
from src.logger import logging # Importing logging module from src.logger for logging errors

def error_message_detail(error,error_detail:sys):
    """
    Extracts detailed error message including the script name, line number, and error description.

    Parameters:
    error (Exception): The error that occurred.
    error_detail (sys): The sys module to extract exception details.

    Returns:
    str: Formatted error message containing script name, line number, and error message.
    """
    _,_,exc_tb = error_detail.exc_info() # Extracts traceback object of the exception
    file_name = exc_tb.tb_frame.f_code.co_filename  # Gets the file name where the error occurred

    # Formatting the error message with script name, line number, and error details
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

# a

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    

