import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Configuración de la página web
st.set_page_config(page_title="App de Optimización", layout="wide")

st.title("🧮 Aplicación de Métodos de Optimización presentado por Los Industrialinas")
st.markdown("### **Integrantes:**)
st.markdown("Catalina Olea") 
st.markdown("Maite Martínez)
st.markdown("Maximiliano Ancán")
st.markdown("Profesor: Gerardo Silva")
st.markdown("Proyecto Final - Métodos de Optimización 2026")
st.markdown("---")

# --- MOTOR MATEMÁTICO DE OPTIMIZACIÓN ---

def evaluar_funcion(func_sympy, variables, punto):
    """Evalúa la función en un punto específico."""
    dicc = {var: val for var, val in zip(variables, punto)}
    return float(func_sympy.subs(dicc))

def calcular_gradiente(func_sympy, variables, punto):
    """Calcula el gradiente numérico en un punto."""
    dicc = {var: val for var, val in zip(variables, punto)}
    grad = [float(sp.diff(func_sympy, var).subs(dicc)) for var in variables]
    return np.array(grad)

def calcular_hessiana(func_sympy, variables, punto):
    """Calcula la matriz Hessiana en un punto (para el método de Newton)."""
    n = len(variables)
    hess = np.zeros((n, n))
    dicc = {var: val for var, val in zip(variables, punto)}
    for i in range(n):
        for j in range(n):
            derivada_2da = sp.diff(sp.diff(func_sympy, variables[i]), variables[j])
            hess[i, j] = float(derivada_2da.subs(dicc))
    return hess

def busqueda_linea_wolfe(f, grad_f, f_sym, vars_sym, x_k, d_k, c1, c2):
    """Encuentra el tamaño de paso alpha que cumple las Condiciones de Wolfe."""
    alpha = 1.0
    v_f_k = f(f_sym, vars_sym, x_k)
    v_grad_k = grad_f(f_sym, vars_sym, x_k)
    
    max_iter_linea = 50
    for _ in range(max_iter_linea):
        x_sig = x_k + alpha * d_k
        v_f_sig = f(f_sym, vars_sym, x_sig)
        v_grad_sig = grad_f(f_sym, vars_sym, x_sig)
        
        # 1ra Condición: Condición de Armijo
        condicion_1 = v_f_sig <= v_f_k + c1 * alpha * np.dot(v_grad_k, d_k)
        
        # 2da Condición: Condición de Curvatura
        condicion_2 = np.dot(v_grad_sig, d_k) >= c2 * np.dot(v_grad_k, d_k)
        
        if condicion_1 and condicion_2:
            return alpha
        
        if not condicion_1:
            alpha *= 0.5
        else:
            alpha *= 2.0
            if alpha > 10.0:
                break
    return alpha

def optimizar(metodo, f_sym, vars_sym, x0, max_iter, tol, c1, c2):
    """Ejecuta el método de optimización seleccionado."""
    historial_puntos = [np.array(x0, dtype=float)]
    x_k = np.array(x0, dtype=float)
    
    d_ant = None
    g_ant = None
    
    for i in range(max_iter):
        g_k = calcular_gradiente(f_sym, vars_sym, x_k)
        norma_g = np.linalg.norm(g_k)
        
        if norma_g < tol:
            break
            
        if metodo == "Método del Gradiente":
            d_k = -g_k
            
        elif metodo == "Método de Newton":
            H_k = calcular_hessiana(f_sym, vars_sym, x_k)
            try:
                d_k = np.linalg.solve(H_k, -g_k)
            except np.linalg.LinAlgError:
                d_k = -g_k
                
        elif metodo == "Gradiente Conjugado":
            if i == 0 or d_ant is None:
                d_k = -g_k
            else:
                beta = np.dot(g_k, g_k) / max(np.dot(g_ant, g_ant), 1e-10)
                d_k = -g_k + beta * d_ant
                if np.dot(d_k, g_k) > 0:
                    d_k = -g_k
            d_ant = d_k.copy()
            g_ant = g_k.copy()
            
        alpha = busqueda_linea_wolfe(evaluar_funcion, calcular_gradiente, f_sym, vars_sym, x_k, d_k, c1, c2)
        
        x_k = x_k + alpha * d_k
        historial_puntos.append(x_k.copy())
        
    return x_k, historial_puntos, i + 1, norma_g

# --- INTERFAZ DE USUARIO ---

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.header("📋 Parámetros del Modelo")
    
    # Control deslizante de 1 a 100 variables
    num_vars = st.slider("Número de variables", min_value=1, max_value=100, value=2)
    metodo = st.selectbox("Método de optimización", ["Método del Gradiente", "Gradiente Conjugado", "Método de Newton"])
    
    # Sugerencias automáticas de funciones
    if num_vars == 1:
        funcion_sug = "x**2 + 4*x + 4"
    elif num_vars == 2:
        funcion_sug = "x**2 + y**2 - 2*x - 4*y"
    elif num_vars == 3:
        funcion_sug = "x**2 + y**2 + z**2 - 2*x + 4*y - 6*z"
    else:
        # Para más de 3 variables, sugerimos un formato indexado x0, x1, x2...
        funcion_sug = " + ".join([f"x{i}**2" for i in range(num_vars)])
        
    func_input = st.text_input("Función objetivo (Operadores matemáticos para la app: * -> Multiplicación, ** -> Exponente)", value=funcion_sug)
    
    # Generación dinámica de los puntos de partida según las variables elegidas
    st.subheader("📍 Punto inicial de partida")
    punto_partida = []
    
    if num_vars == 1:
        x0 = st.number_input("x0", value=2.0)
        punto_partida = [x0]
    elif num_vars == 2:
        x0 = st.number_input("x0", value=4.0)
        y0 = st.number_input("y0", value=4.0)
        punto_partida = [x0, y0]
    elif num_vars == 3:
        x0 = st.number_input("x0", value=4.0)
        y0 = st.number_input("y0", value=4.0)
        z0 = st.number_input("z0", value=4.0)
        punto_partida = [x0, y0, z0]
    else:
        # Si eligen de 4 a 100 variables, creamos una lista dinámica de entradas
        for i in range(num_vars):
            val = st.number_input(f"Variable x{i}", value=2.0, key=f"var_{i}")
            punto_partida.append(val)
        
    st.subheader("⚙️ Parámetros de Control")
    max_iter = st.number_input("Número máximo de iteraciones", value=100, step=10)
    tol = st.number_input("Tolerancia de convergencia", value=1e-5, format="%.5f")
    
    st.subheader("🔍 Condiciones de Wolfe")
    c1 = st.number_input("Parámetro c1 (Armijo)", value=1e-4, format="%.4f")
    c2 = st.number_input("Parámetro c2 (Curvatura)", value=0.9, format="%.2f")

with col_der:
    st.header("📊 Resultados de la búsqueda")
    
    if st.button(" 📝 Buscar la optimización", type="primary"):
        try:
            # Definir símbolos matemáticos dinámicamente según la cantidad elegida
            if num_vars == 1:
                vars_sym = [sp.Symbol('x')]
            elif num_vars == 2:
                vars_sym = list(sp.symbols('x y'))
            elif num_vars == 3:
                vars_sym = list(sp.symbols('x y z'))
            else:
                vars_sym = [sp.Symbol(f'x{i}') for i in range(num_vars)]
                
            f_sym = sp.sympify(func_input)
            
            # Ejecutar optimización numérica
            min_encontrado, historial, iters, error_f = optimizar(
                metodo, f_sym, vars_sym, punto_partida, max_iter, tol, c1, c2
            )
            
            valor_final_f = evaluar_funcion(f_sym, vars_sym, min_encontrado)
            
            # Formatear el texto de salida del punto mínimo de forma limpia
            texto_coordenadas = " , ".join([f"{val:.4f}" for val in min_encontrado])
            st.success(f"🎯 Punto mínimo encontrado en coordenadas: **({texto_coordenadas})**")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric(label="Valor mínimo f(x*)", value=f"{valor_final_f:.5f}")
            c_m2.metric(label="Iteraciones realizadas", value=str(iters))
            c_m3.metric(label="Error final (||∇f||)", value=f"{error_f:.5e}")
            
            # --- SECCIÓN DE GRÁFICOS REPARADA ---
            historial = np.array(historial)
            
            if num_vars == 1:
                st.subheader("📈 Gráfico de Convergencia 2D")
                fig, ax = plt.subplots(figsize=(8, 4))
                
                x_min, x_max = historial[:, 0].min() - 2, historial[:, 0].max() + 2
                X_vals = np.linspace(x_min, x_max, 200)
                Y_vals = [evaluar_funcion(f_sym, vars_sym, [val]) for val in X_vals]
                
                ax.plot(X_vals, Y_vals, 'b-', label='f(x)')
                Y_hist = [evaluar_funcion(f_sym, vars_sym, [val]) for val in historial[:, 0]]
                ax.plot(historial[:, 0], Y_hist, 'r.-', label='Trayectoria del algoritmo')
                ax.plot(min_encontrado[0], valor_final_f, 'go', label=f'Mínimo: {min_encontrado[0]:.2f}')
                
                ax.set_xlabel('Variable X')
                ax.set_ylabel('f(x)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
                
            elif num_vars == 2:
                st.subheader("Gráfico de Superficie 3D y Mínimo Encontrado")
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                x_min, x_max = min(historial[:, 0].min() - 1, punto_partida[0] - 1), max(historial[:, 0].max() + 1, punto_partida[0] + 1)
                y_min, y_max = min(historial[:, 1].min() - 1, punto_partida[1] - 1), max(historial[:, 1].max() + 1, punto_partida[1] + 1)
                
                X, Y = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))
                Z = np.zeros_like(X)
                for i_m in range(X.shape[0]):
                    for j_m in range(X.shape[1]):
                        Z[i_m, j_m] = evaluar_funcion(f_sym, vars_sym, [X[i_m, j_m], Y[i_m, j_m]])
                
                superficie = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
                
                # Camino e indicadores del mínimo en la superficie
                z_historial = [evaluar_funcion(f_sym, vars_sym, p) for p in historial]
                ax.plot(historial[:, 0], historial[:, 1], z_historial, 'r.-', label='Camino de convergencia', markersize=6)
                ax.scatter(min_encontrado[0], min_encontrado[1], valor_final_f, color='rose', s=200, marker='*', label='Mínimo exacto', depthshade=False)
                
                ax.set_xlabel('Eje X')
                ax.set_ylabel('Eje Y')
                ax.set_zlabel('f(X, Y)')
                ax.legend()
                fig.colorbar(superficie, ax=ax, shrink=0.5, aspect=5)
                st.pyplot(fig)
                
            else:
                # Alerta informativa para 3 o más variables (donde no se puede graficar el espacio físico completo)
                st.info("ℹ️ El cálculo matemático se ha realizado con éxito. Recuerda que la visualización gráfica de funciones solo está disponible para modelos de 1 y 2 variables físicos.")
                
        except Exception as e:
            st.error(f"⚠️ Error al interpretar la función matemática o en los parámetros: {e}")
