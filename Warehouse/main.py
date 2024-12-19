from WarehouseAdmin import WarehouseAdmin
from WarehouseGUI import WarehouseGUI

def main():
    warehouse_admin = WarehouseAdmin()
    gui = WarehouseGUI(warehouse_admin)
    gui.run()

if __name__ == "__main__":
    main()
