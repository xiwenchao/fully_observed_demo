class message():
    '''
    The data structure for message.
    We might need to add some when it comes to conversational.
    '''

    def __init__(self, sender, receiver, message_type, data):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.data = data


