import joblib
import subprocess

model_path = "btree.joblib"
model = joblib.load(model_path)
tree = model.tree_

n_nodes = tree.node_count
children_left = tree.children_left
children_right = tree.children_right
feature = tree.feature
threshold = tree.threshold
values = tree.value.squeeze()

includes = "#include <stdio.h>"
c_nodes = f"""
typedef struct {{
    int feature;
    float threshold;
    int left;
    int right;
    float value;
}} Node;

Node tree[{n_nodes}] = {{
{",\n".join([f"""{{ {feature[i]},
{threshold[i]:.6f}f,
{children_left[i]},
{children_right[i]},
{float(values[i]) if values.ndim == 1 else float(values[i,1]):.6f} }}"""
for i in range(n_nodes)])}
}};
"""

c_function = """
float predict_tree(float* features) {
    int node = 0;
    while (tree[node].feature != -2) {
        if (features[tree[node].feature] <= tree[node].threshold)
            node = tree[node].left;
        else
            node = tree[node].right;
    }
    return tree[node].value;
}
"""

c_main = """
int main() {
    float X[] = {0.0f, 1.0f, 0.0f};  // Exemple
    float y_pred = predict_tree(X);
    printf("Prediction = %f\\n", y_pred);
    return 0;
}
"""

output_c_file = "model_inference.c"
with open(output_c_file, "w") as f:
    f.write(includes)
    f.write(c_nodes)
    f.write(c_function)
    f.write(c_main)

# Compilation
output_exe = "model_inference"
compile_cmd = f"gcc {output_c_file} -o {output_exe}"

try:
    subprocess.run(compile_cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print("Compilation error:", e)
    exit(1)