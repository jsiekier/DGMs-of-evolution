from data_processing.python_doc_reader_writer import read_python_dict_save, write_python


def process_solution_file(file_name,fix_variables):

    with open(file_name, 'r') as file:
        for line in file:
            parts = line.strip().split("\t")  # Split by tab

            if len(parts) < 3:
                continue  # Skip malformed lines

            var0 = parts[0].split(";") if parts[0] != "NA" else []  # Convert to list, handle NA
            variable1 = parts[1]
            variable2 = parts[2]

            if variable2 not in fix_variables:
                fix_variables[variable2] = {'Nes': [], 'Ne': []}

            fix_variables[variable2]['Nes'] = [float(x) for x in var0 if x]  # Convert values to float
            fix_variables[variable2]['Ne'] = [float(variable1)]  # Store variable1 as a list

    return fix_variables

if __name__ == '__main__':


    python_file_name = "variables.py"
    fix_variables, var_name = read_python_dict_save(python_file_name)
    # Print results
    print(type(fix_variables))  # Should be <class 'dict'>
    # read new data entry:
    solution_file='solution_tmp.txt'

    # extend dict:
    fix_variables=process_solution_file(solution_file,fix_variables)

    # write added dict to the python file:
    write_python(python_file_name, var_name, fix_variables)