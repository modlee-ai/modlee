#%%
import torch
import torch.onnx
try:
    from pytorch_forecasting import models as pfm
    from pytorch_forecasting import NBeats, AutoRegressiveBaseModel, DecoderMLP
    import pandas as pd
    from lightning.pytorch import Trainer
    from modlee.converter import Converter
    import onnx_graphsurgeon as gs
    converter = Converter()
    # Load data
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    from modlee.timeseries_dataloader import TimeseriesDataset
    import torch.onnx

    dataset = TimeseriesDataset(data=data, encoder_column=encoder_column, target = 'Close', time_column = 'Date', input_seq=2, output_seq=1)
    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.01,
        limit_train_batches=150,
    )
    model = NBeats.from_dataset(
        dataset=dataset.get_dataset(), 
    )
    from modlee.data_metafeatures import TimeseriesDataMetafeatures
    meta = TimeseriesDataMetafeatures(dataset.to_dataloader())
    features = meta.calculate_metafeatures()
    #meta.print_meta(features=features)

    #trainer.fit(model=model, train_dataloaders=dataset.to_dataloader())

    model.eval()
    sample_data = next(iter(dataset.to_dataloader(batch_size=1)))

    x = {'x':sample_data[0]}


    # with open('input_file2.txt', 'r') as f:
    #     onnx_text = f.read()
    #     f.close()

    #onnx_graph = converter.torch_model2onnx_graph(model, input_dummy=x)
    #onnx_text = converter.format_onnx_text(converter.onnx_graph2onnx_text(onnx_graph))
    # print("Converted from graph to text")
    # print(onnx_text)
    # onnx_text = converter.format_onnx_text(onnx_text)
    #onnx_graph = converter.onnx_text2torch_code(onnx_text)
    #onnx_graph = converter.onnx_text2onnx_graph(onnx_text)
    # print("Converted from text to graph")
    # with open('output_file2.txt', 'w') as f:
    #     f.write(onnx_text)
    #     f.close()
    # onnx_graph = gs.import_onnx(onnx_graph)
    # print("Imported onnx graph")

    # onnx_graph = converter.init_graph_tensors(onnx_graph)
    # print("Initialized graph tensors")

    # onnx_graph = gs.export_onnx(onnx_graph)
    # print("Exported onnx graph")

    # torch_model = converter.onnx_graph2torch_model(onnx_graph)


    # ### routine of pytest
    onnx_graph = converter.torch_model2onnx_graph(model, input_dummy=x)
    torch_model = converter.onnx_graph2torch_model(onnx_graph)
    onnx_text = converter.onnx_graph2onnx_text(onnx_graph)
    onnx_graph = converter.onnx_text2onnx_graph(onnx_text)
    onnx_text = converter.format_onnx_text(onnx_text)
    print(onnx_text)
    torch_code = converter.onnx_text2torch_code(onnx_text)
    torch_model = converter.torch_code2torch_model(torch_code)
    print("Conversion pipeline passed")
except:
    pass