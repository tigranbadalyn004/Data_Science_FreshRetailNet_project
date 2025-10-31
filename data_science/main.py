# main.py
import src.data.data_loader as dl

data_loading = dl.FreshRetailDataLoader()

data_loading.load_data()
print(data_loading.get_data_summary())

