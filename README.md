# ClosedGPT

ClosedGPT is a question & answer system for your PDF/CSV or any local documents, 100% offline with no data leaks! 🇮🇳
## Installation

Start with environment setup as given below
Install conda

```shell
conda create -n localGPT
```

Activate

```shell
conda activate localGPT
```

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

## Usage
Put any and all of your .txt, .pdf, or .csv files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory.

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.

Run the following command to ingest all the data.

`defaults to cuda`

```shell
python ingest.py
``````

For Ingestion run the following:

```shell
python ingest.py --device_type cpu
```

In order to ask a question, run a command like:

```shell
python run_localGPT.py --device_type cpu
```

## For UI

1. Run the following command `python run_localGPT_API.py`. The API should being to run.

2. Wait until everything has loaded in. You should see something like `INFO:werkzeug:Press CTRL+C to quit`.

3. Open up a second terminal and activate the same python environment.

4. Navigate to the `/LOCALGPT/localGPTUI` directory.

5. Run the command `python localGPTUI.py`.

6. Open up a web browser and go the address `http://localhost:5111/`.


## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)