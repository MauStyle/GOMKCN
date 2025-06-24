import subprocess
command = ['python', 'main.py']

for epochs in [200]:
    for lr in [0.01]:
        for batch_size in [128]:
            for dataset in ['ENZYMES']:
                for fold in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    for k_hop in [2]:
                        for s_subgraph in [14]:
                            for n_filter in [[[32, 16]]]:
                                for k_step in [2]:
                                    for tao in [0.05]:
                                        for norm in ['True']:
                                            for relu in ['True']:
                                                for pool in ['mean']:
                                                    params = ['-epochs', str(epochs),
                                                              '-lr', str(lr),
                                                              '-batch_size', str(batch_size),
                                                              '-dataset', str(dataset),
                                                              '-fold', str(fold),
                                                              '-k_hop', str(k_hop),
                                                              '-s_subgraph', str(s_subgraph),
                                                              '-n_filter', str(n_filter),
                                                              '-k_step', str(k_step),
                                                              '-tao', str(tao),
                                                              '-norm', str(norm),
                                                              '-relu', str(relu),
                                                              '-pool', str(pool)
                                                              ]
                                                    full_command = command + params
                                                    print(f"Running: {' '.join(full_command)}")
                                                    result = subprocess.run(full_command, check=False, capture_output=False, text=False)
                                                    print(f"Output for"
                                                          f"epochs={epochs}, "
                                                          f"lr={lr}, "
                                                          f"batch_size={batch_size}, "
                                                          f"dataset={dataset}, "
                                                          f"fold={fold}, "
                                                          f"k_hop={k_hop}, "
                                                          f"s_subgraph={s_subgraph}, "
                                                          f"n_filter={n_filter}, "
                                                          f"k_step={k_step}, "
                                                          f"tao={tao}, "
                                                          f"norm={norm}, "
                                                          f"relu={relu}, "
                                                          f"pool={pool}, "
                                                          f":\n{result.stdout}")
                                                    if result.stderr:
                                                        print(f"Output for"
                                                              f"epochs={epochs}, "
                                                              f"lr={lr}, "
                                                              f"batch_size={batch_size}, "
                                                              f"dataset={dataset}, "
                                                              f"fold={fold}, "
                                                              f"k_hop={k_hop}, "
                                                              f"s_subgraph={s_subgraph}, "
                                                              f"n_filter={n_filter}, "
                                                              f"k_step={k_step}, "
                                                              f"tao={tao}, "
                                                              f"norm={norm}, "
                                                              f"relu={relu}, "
                                                              f"pool={pool}, "
                                                              f":\n{result.stderr}")

print("All runs completed.")


