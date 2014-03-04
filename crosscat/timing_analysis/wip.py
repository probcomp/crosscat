import crosscat.utils.timing_test_utils as ttu


all_kernels = [
    'column_partition_hyperparameter',
    'column_partition_assignments',
    'column_hyperparameters',
    'row_partition_hyperparameters',
    'row_partition_assignments',
    ]


if __name__ == '__main__':
    configs = ttu.gen_configs(
            kernel_list=[[kernel] for kernel in all_kernels],
            num_rows=[10, 100],
            )
    results = map(ttu.runner, configs)
    for el in zip(configs, results):
        print; print el; print

