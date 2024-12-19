class OrderProcessor:
    def __init__(self, warehouse_layout):
        self.warehouse_layout = warehouse_layout

    def process_order(self, order):
        processed_items = []

        # Sort items in the order based on priority (or any other relevant criterion)
        sorted_items = sorted(order.items, key=lambda item_id: self.warehouse_layout.get_item_priority(item_id))

        for item_id in sorted_items:
            nearest_items = self.warehouse_layout.find_nearest_items(item_id)
            
            # Basic logic: pick the first nearest item
            if nearest_items:
                processed_items.append(nearest_items[0])

        return processed_items
      
    def process_order(self, order):
        """
        Processes an order using a basic Greedy Algorithm.
        """
        processed_items = []
        for item_id in order.items:
            nearest_items = self.warehouse_layout.find_nearest_items(item_id)
            # Basic logic: pick the first nearest item
            if nearest_items:
                processed_items.append(nearest_items[0])
        return processed_items

    def prioritize_orders(self, orders):
        """
        Prioritizes orders based on total weight (heavier orders first).
        """
        return sorted(orders, key=lambda o: sum(item.weight for item in o.items), reverse=True)