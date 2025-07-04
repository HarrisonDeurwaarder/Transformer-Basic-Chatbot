import time


def timer(func):
    '''
    Times a function call and displays the result
    '''
    def wrapper(*args, **kwargs):
        # Get time before function call
        start = time.time()
        result = func(*args, **kwargs)
        # Print result
        print(f'\033[90m{func.__name__.title()} took {time.time() - start:.5f} seconds.\033[0m')
        return result
    return wrapper