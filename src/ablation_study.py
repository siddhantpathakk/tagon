import json
from trainer import Trainer
from utils.data_utils import Data
from utils.main_utils import parse_opt, set_up_logger

list_n_degree = [0, 10, 20, 30]
list_n_head = [1,2,4]
list_n_layer = [0,1,2,4]
list_node_dim = [8,16,32] # 64 avoided due to CUDA memory error
list_time_dim = [8,16,32] # 64 avoided due to CUDA memory error
list_agg_method = ['attn', 'lstm', 'mean']
list_attn_mode = ['prod', 'map']
list_time = ['time', 'pos', 'empty', 'disentangle']
list_neg_sample_eval = [0, 1, 10, 100]

list_lr = [1e-2, 1e-3]
list_l2 = [1e-1, 1e-2, 1e-3]

param_grid = {
    'n_head': list_n_head,
    'n_layer': list_n_layer,
    'node_dim': list_node_dim,
    'time_dim': list_time_dim,
    'agg_method': list_agg_method,
    'attn_mode': list_attn_mode,
    'time': list_time,
    'neg_sample_eval': list_neg_sample_eval,
    'lr': list_lr,
    'l2': list_l2
}

class AblationStudy:
    def __init__(self, args, param_grid):
        self.args = args
        assert self.args.pretrain is not None, 'Pretrained model path is required for ablation study.'
        self.args.n_epoch = 10
        self.logger = set_up_logger()
        
        self.data = Data(args.data, args)
        
        self.param_grid = param_grid
        
        self.outcomes = {}
        self.output_file = '/home/FYP/siddhant005/fyp/log/ablation_study/ablation_study_results.json'
      
        self.logger.info(args)
        
    def run_parametric(self, param_to_test, param_values):
        final_test_results = {}
        for value in param_values:
            final_test_results[value] = {}
            self.logger.info(f'Commencing training for {param_to_test} = {value}')
            setattr(self.args, param_to_test, value)
            trainer = Trainer(self.args, self.data, self.logger)
            trainer.train()
            
            for key in ['train_loss','test_recall10', 'test_recall20', 'test_mrr']:
                final_test_results[value][key] = trainer.history[key]
            
            self.logger.info(f'Training finished for {param_to_test} = {value}.')
            
        self.logger.info(f'Final test results for {param_to_test}: {final_test_results}')
        return final_test_results
        
        
    def run(self):
        for param, values in self.param_grid.items():
            results_parameter = self.run_parametric(param, values)
            self.outcomes[param] = results_parameter
    
        self.export_results()
        self.logger.info(f'Completed ablation study. Results exported to {self.output_file}.')
        
        
    def export_results(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.outcomes, f)
        self.logger.info('Results exported to ablation_study_results.json')
        
        
if __name__ == '__main__':
    args = parse_opt()
    ablation_study = AblationStudy(args, param_grid)
    ablation_study.run()
    