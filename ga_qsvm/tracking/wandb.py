def build_train_wandb_config(dataset_name, params, machine_id, index):
    kernel = params.get("kernel", "pqk")
    return {
        "project": f"GA-QSVM-{dataset_name}-{kernel}-N{params['num_qubits']}-D{params['depth']}-C{params['num_circuit']}",
        "name": f"n{params['num_qubits']}-c{params['num_cnot']}-D{params['depth']}-C{params['num_circuit']}-g{params['num_generation']}-p{round(params['prob_mutate'], 5)}-id{machine_id}",
        "config": {**params, "i": index},
    }
