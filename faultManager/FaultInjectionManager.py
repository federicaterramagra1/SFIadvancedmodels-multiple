import itertools  # Add this import

class FaultInjectionManager:
    def __init__(self,
                 network: Module,
                 network_name: str,
                 device: torch.device,
                 loader: DataLoader,
                 clean_output: torch.Tensor,
                 injectable_modules: List[Union[Module, List[Module]]] = None,
                 num_faults_to_inject: int = 1):  # Number of faults to inject simultaneously
        self.network = network
        self.network_name = network_name
        self.loader = loader
        self.device = device

        self.clean_output = clean_output
        self.faulty_output = list()

        self.num_faults_to_inject = num_faults_to_inject  # Number of faults to inject

        # Other initializations remain the same
        self.__log_folder = f'log/{self.network_name}/batch_{self.loader.batch_size}'
        self.__faulty_output_folder = SETTINGS.FAULTY_OUTPUT_FOLDER
        self.skipped_inferences = 0
        self.total_inferences = 0
        self.weight_fault_injector = WeightFaultInjector(self.network)
        self.injectable_modules = injectable_modules

    def run_faulty_campaign_on_weight(self,
                                      fault_model: str,
                                      fault_list: list,
                                      first_batch_only: bool = False,
                                      force_n: int = None,
                                      save_output: bool = False,
                                      save_ofm: bool = False,
                                      ofm_folder: str = None) -> (str, int):
        """
        Run a faulty injection campaign for the network with support for multiple faults injection (1, 2, or 3).
        """
        self.skipped_inferences = 0
        self.total_inferences = 0

        total_different_predictions = 0
        total_predictions = 0

        average_memory_occupation = 0
        total_iterations = 1

        with torch.no_grad():
            if force_n is not None:
                fault_list = fault_list[:force_n]

            # Generate combinations of faults based on num_faults_to_inject
            fault_combinations = list(itertools.combinations(fault_list, self.num_faults_to_inject))

            start_time = time.time()

            accuracy_dict = dict()

            # Loop over batches in the data loader
            for batch_id, batch in enumerate(self.loader):
                data, target = batch
                data = data.to(self.device)

                accuracy_batch_dict = dict()
                accuracy_dict[batch_id] = accuracy_batch_dict

                faulty_prediction_dict = dict()
                batch_clean_prediction_scores = [float(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).values]
                batch_clean_prediction_indices = [int(fault) for fault in torch.topk(self.clean_output[batch_id], k=1).indices]

                # Inject multiple faults in a single batch
                pbar = tqdm(fault_combinations,
                            colour='green',
                            desc=f'FI on b {batch_id}',
                            ncols=shutil.get_terminal_size().columns * 2)
                for fault_combination in pbar:
                    # Inject all faults in the combination
                    for fault in fault_combination:
                        if fault_model == 'byzantine_neuron':
                            injected_layer = self.__inject_fault_on_neuron(fault=fault)
                        elif fault_model == 'stuck-at_params':
                            self.__inject_fault_on_weight(fault, fault_mode='stuck-at')
                        else:
                            raise ValueError(f'Invalid fault model {fault_model}')

                    # Measure memory usage
                    torch.cuda.reset_peak_memory_stats()

                    # Save IFM if required
                    if save_ofm:
                        for injectable_module in self.injectable_modules:
                            injectable_module.ifm_path = f'{ofm_folder}/fault_{fault_combination}_batch_{batch_id}_layer_{injectable_module.layer_name}'

                    # Run inference with faults injected
                    faulty_scores, faulty_indices, different_predictions = self.__run_inference_on_batch(batch_id=batch_id,
                                                                                                         data=data)

                    # Handle faulty predictions
                    if faulty_indices is None:
                        faulty_scores = self.clean_output[batch_id]
                        faulty_indices = batch_clean_prediction_indices

                    # Calculate accuracy for the batch
                    accuracy_batch_dict[str(fault_combination)] = float(torch.sum(target.eq(torch.tensor(faulty_indices)) / len(target))

                    # Store faulty predictions
                    faulty_scores = faulty_scores.detach().cpu()
                    faulty_prediction_dict[str(fault_combination)] = tuple(zip(faulty_indices, faulty_scores))
                    total_different_predictions += different_predictions

                    # Save output if required
                    if save_output:
                        self.faulty_output.append(faulty_scores.numpy())

                    # Update progress
                    total_predictions += len(batch[0])
                    different_predictions_percentage = 100 * total_different_predictions / total_predictions
                    pbar.set_postfix({'Different': f'{different_predictions_percentage:.6f}%',
                                      'Skipped': f'{100 * self.skipped_inferences / self.total_inferences:.2f}%',
                                      'Avg. memory': f'{average_memory_occupation} MB'}
                                     )

                    # Clean up injected faults
                    if fault_model == 'byzantine_neuron':
                        injected_layer.clean_fault()
                    elif fault_model == 'stuck-at_params':
                        self.weight_fault_injector.restore_golden()
                    else:
                        raise ValueError(f'Invalid fault model {fault_model}')

                    total_iterations += 1

                # Log accuracy metrics for the batch
                os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
                log_filename = f'{self.__log_folder}/{fault_model}/batch_{batch_id}.csv'
                with open(log_filename, 'w') as log_file:
                    log_writer = csv.writer(log_file)
                    log_writer.writerows(accuracy_batch_dict.items())

                # Save faulty output if required
                if save_output:
                    os.makedirs(f'{self.__faulty_output_folder}/{fault_model}', exist_ok=True)
                    np.save(f'{self.__faulty_output_folder}/{fault_model}/batch_{batch_id}', self.faulty_output)
                    self.faulty_output = list()

                # Stop after the first batch if required
                if first_batch_only:
                    break

        # Calculate average accuracy
        average_accuracy_dict = dict()
        for fault_combination in fault_combinations:
            fault_accuracy = np.average([accuracy_batch_dict[str(fault_combination)] for _, accuracy_batch_dict in accuracy_dict.items()])
            average_accuracy_dict[str(fault_combination)] = float(fault_accuracy)

        # Log final results
        os.makedirs(f'{self.__log_folder}/{fault_model}', exist_ok=True)
        log_filename = f'{self.__log_folder}/{fault_model}/all_batches.csv'
        with open(log_filename, 'w') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerows(average_accuracy_dict.items())

        elapsed = math.ceil(time.time() - start_time)
        return str(timedelta(seconds=elapsed)), average_memory_occupation