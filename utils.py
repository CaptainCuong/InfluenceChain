def distributed_log(args, message):
    """
    Log a message only from the main process in distributed training.
    
    Args:
        args: Argument parser containing distributed settings.
        message (str): The message to log.
    """
    if args.local_rank == 0:
        print(message)