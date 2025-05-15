import pprint
import re
import ast


def write_python(python_file_name, var_name, fix_variables):
    print(var_name)
    with open(python_file_name, "w") as f:
        f.write(f"{var_name} = ")
        f.write(pprint.pformat(fix_variables, indent=4))
        f.write("\n")  # Ensure a newline at the end


def read_python_dict_save(python_file_name):
    filtered_lines = []
    inside_multiline_comment = False
    with open(python_file_name, "r") as f:
        for line in f:
            line = line.strip()

            # Handle multi-line comments ''' or """
            if line.startswith(("'''", '"""')):
                inside_multiline_comment = not inside_multiline_comment
                continue
            if inside_multiline_comment:
                continue

            # Ignore full-line comments
            if line.startswith("#") or not line:
                continue

            # Remove inline comments (anything after #)
            line = re.split(r'\s+#', line, 1)[0]  # Keep only code before #

            filtered_lines.append(line)

    # Join the filtered lines back into a single string
    cleaned_content = "\n".join(filtered_lines)

    # Extract dictionary by removing variable name (e.g., fix_variables=)
    var_name=''
    if "=" in cleaned_content:
        var_name = cleaned_content.split("=", 1)[0].strip()
        cleaned_content = cleaned_content.split("=", 1)[1].strip()


    # Convert the dictionary string into an actual dictionary
    #print(cleaned_content)
    fix_variables = ast.literal_eval(cleaned_content)
    return fix_variables,var_name
