def build_train_wandb_config(dataset_name, params, machine_id, index):
    return {
        "project": f"GA-QSVM-{dataset_name}-N{params['num_qubits']}-D{params['depth']}-C{params['num_circuit']}",
        "name": (
            f"x{params['num_rx']}-y{params['num_ry']}-z{params['num_rz']}"
            f"-c{params['num_circuit']}-g{params['num_generation']}"
            f"-p{round(params['prob_mutate'], 5)}-id{machine_id}"
        ),
        "config": {**params, "i": index},
    }
