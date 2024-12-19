class WarehouseLayout:
    def __init__(self):
        self.items = {}  # key: Product_ID, value: WarehouseItem

    def add_item(self, item):
        self.items[item.product_id] = item

    def get_item(self, product_id):
        return self.items.get(product_id)
    
    def optimize_layout(self):
        """
        Reorganizes items based on predefined optimization criteria.
        """
        # Example logic: sort items by weight (heaviest to lightest)
        sorted_items = dict(sorted(self.items.items(), key=lambda x: x[1].weight, reverse=True))
        self.items = sorted_items

    def find_nearest_items(self, product_id):
        """
        Finds and returns items nearest to a given product ID.
        Placeholder logic: return items with similar weights.
        """
        target_item = self.items.get(product_id)
        if not target_item:
            return []
        similar_items = [item for item in self.items.values() if abs(item.weight - target_item.weight) < 5]
        return similar_items
