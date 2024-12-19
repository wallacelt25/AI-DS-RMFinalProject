class WarehouseItem:
    def __init__(self, product_id, location, weight, priority, size):
        self.product_id = product_id
        self.location = location
        self.weight = float(weight)
        self.priority = self.map_priority(priority)
        self.size = size

    def map_priority(self, priority_str):
        priority_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
        return priority_mapping.get(priority_str.strip(), 1)

    # Additional methods as needed
