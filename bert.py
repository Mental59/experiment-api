from app.services.experiment.transformers import test
from huggingface_hub import HfApi, utils


def main():
    # train.run()
    model_name_or_path = 'dslim/bert-base-NER34'
    api = HfApi()
    try:
        model_info = api.model_info(model_name_or_path)
    except utils.HfHubHTTPError as exc:
        print(exc.server_message)
        return

    test.run_by_model_name_or_path(
        project='TestExperiments',
        run_name='Test BERT 1',
        dataset='menu_txt_no_tags',
        experiment_tracker_type='mlflow',
        model_name_or_path=model_name_or_path,
        task='ner'
    )


if __name__ == '__main__':
    main()
