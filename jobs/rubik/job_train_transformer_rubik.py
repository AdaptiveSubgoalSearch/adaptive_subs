from jobs.job_train_transformer import HfTrainingPipeline
from utils import hf as hf_utils
from utils import hf_generate


class HfTrainingPipelineRubik(HfTrainingPipeline):
    def _generate_sample_sequences(self, model, eval_datasets, done_epochs):
        for dataset, metric_key_prefix in eval_datasets:
            sequences, scores = hf_generate.generate_sequences(
                model=model,
                inputs=[hf_generate.GenerationInput(
                input_ids=entry['input_ids'],
                          attention_mask=entry['attention_mask'],
                ) for entry in dataset][:100],
                num_return_sequences=3,
                num_beams=5,
                max_length=57,
            )
            hf_utils.log_prediction_triple(
                input_ids=dataset[0]['input_ids'],
                target_ids=dataset[0]['labels'],
                predictions_ids=sequences[0][:1],
                tokenizer=self.tokenizer,
                log_prefix=f'generate_{metric_key_prefix}'
            )
            sequences = [self.tokenizer.decode(sequence[0])[1:] for sequence in sequences]
            print(metric_key_prefix, sequences)
            self._log_validity_metrics(dataset, sequences, self.tokenizer, done_epochs, 'generate_' + metric_key_prefix)

    def _log_validity_metrics(self, dataset, sequences, tokenizer, done_epochs, prefix, model=None):
        raise NotImplementedError()