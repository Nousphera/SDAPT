class UShapedClient():
    def __init__(self, client_id, client_head, client_tail, client_optimizer, client_scheduler, client_train_iterator, client_test_dataloader, criterion):
        self.client_id = client_id
        self.head = client_head
        self.tail = client_tail
        self.optimizer = client_optimizer
        self.scheduler = client_scheduler
        self.train_iterator = client_train_iterator
        self.test_dataloader = client_test_dataloader
        self.test_iterator = iter(client_test_dataloader)
        self.criterion = criterion

    def get_loss(self, predictions, y):
        loss = self.criterion(predictions, y)
        return loss