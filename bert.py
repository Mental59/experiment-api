from app.services.experiment.transformers import test
from app.services.experiment.transformers import train


def main():
    # train.run()
    test.run_by_model_name_or_path(
        project='TestExperiments',
        run_name='Test BERT 1',
        dataset='menu_txt_tagged_fixed_bottlesize',
        experiment_tracker_type='mlflow',
        model_name_or_path='google-bert/bert-base-uncased',
        task='ner'
    )


if __name__ == '__main__':
    main()
