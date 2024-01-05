import multiprocessing

def train_model(train_data):
    # Your training code here using train_data
    pass  # Replace with your actual training logic

def test_model(test_data):
    # Your testing/evaluation code here using test_data
    pass  # Replace with your actual testing logic

sorted_list_of_dfs = get_df_pkl_lst()
num_generations = 5  # Replace this with your desired number of generations
window_size = 2  # Define the size of the rolling window

if len(sorted_list_of_dfs) >= window_size:
    for i in range(0, len(sorted_list_of_dfs) - window_size + 1):
        train_df_path = sorted_list_of_dfs[i]
        test_df_path = sorted_list_of_dfs[i + 1]

        train_df = load_dataframe(train_df_path)
        test_df = load_dataframe(test_df_path)

        # Use multiprocessing to run training and testing in separate processes
        train_process = multiprocessing.Process(target=train_model, args=(train_df,))
        test_process = multiprocessing.Process(target=test_model, args=(test_df,))

        # Start training and testing processes
        train_process.start()
        test_process.start()

        # Wait for both processes to finish
        train_process.join()
        test_process.join()

        # Move the window forward
        train_df_path = test_df_path
        test_df_path = sorted_list_of_dfs[i + window_size]
