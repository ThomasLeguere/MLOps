import joblib
import subprocess

# Model parameters
model_path = "regression.joblib"
model = joblib.load(model_path)

thetas = [ f"{model.intercept_:.6f}" ] + [f"{coef:.6f}" for coef in model.coef_]

# C code
includes = """
#include <stdio.h>
"""

c_code_exp_approx = """
float exp_approx(float x, int n_term){
    float res = 0.0;
    float x_pow = 1.0;
    float factorial_i = 1.0;
    
    for (int i = 0; i <= n_term; i++){
        res += x_pow / factorial_i;
        x_pow *= x;
        factorial_i *= i + 1;
    }
    
    return res;
}
"""

c_code_sigmoid = """
float sigmoid(float x){
    return 1 / (1 + exp_approx(-x, 10));
}
"""

c_code_logistic = f"""
float prediction(float *features, int n_features) {{
    float thetas[{len(thetas)}] = {"{" + ", ".join(thetas) + "}"};
    float result = thetas[0];
    for (int i = 0; i < n_features; i++) {{
        result += features[i] * thetas[i + 1];
    }}
    return sigmoid(result);
}}
"""

main = """
int main() {
    float X[] = {1.0f, 2.0f, 3.0f};
    int n_features = sizeof(X) / sizeof(float);

    float y_pred = prediction(X, n_features);
    printf("Prediction = %f\\n", y_pred);
    return 0;
}
"""

# C file creation
output_c_file = "model_inference.c"
with open(output_c_file, "w") as f:
    f.write(includes)
    f.write(c_code_exp_approx)
    f.write(c_code_sigmoid)
    f.write(c_code_logistic)
    f.write(main)

# Compilation
output_exe = "model_inference"
compile_cmd = f"gcc {output_c_file} -o {output_exe}"

try:
    subprocess.run(compile_cmd, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print("Compilation error:", e)
    exit(1)
