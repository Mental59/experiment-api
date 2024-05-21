from app.services.experiment.transformers import test
from app.services.experiment.transformers import train


def main():
    # train.run()
    test.run_by_model_name_or_path(
        project='TestExperiments',
        run_name='Test BERT 1',
        dataset='menu_txt_no_tags',
        experiment_tracker_type='mlflow',
        model_name_or_path='dslim/bert-base-NER',
        task='ner'
    )


if __name__ == '__main__':
    main()
