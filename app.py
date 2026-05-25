import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Configuración de la página web
st.set_page_config(page_title="App de Optimización", layout="wide")

st.title("🧮 Aplicación de Métodos de Optimización")
st.markdown("Proyecto Final - Métodos de Optimización 2026")

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
        
        # 1ra Condición: Condición de Armijo (Decrecimiento suficiente)
        condicion_1 = v_f_sig <= v_f_k + c1 * alpha * np.dot(v_grad_k, d_k)
        
        # 2da Condición: Condición de Curvatura
        condicion_2 = np.dot(v_grad_sig, d_k) >= c2 * np.dot(v_grad_k, d_k)
        
        if condicion_1 and condicion_2:
            return alpha
        
        # Si no cumple, reducimos alpha (búsqueda simple por backtracking hacia atrás/adelante)
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
    
    # Para el método del Gradiente Conjugado necesitamos guardar la dirección anterior
    d_ant = None
    g_ant = None
    
    for i in range(max_iter):
        g_k = calcular_gradiente(f_sym, vars_sym, x_k)
        norma_g = np.linalg.norm(g_k)
        
        if norma_g < tol:
            break
            
        # Definición de la dirección de búsqueda d_k según el método
        if metodo == "Método del Gradiente":
            d_k = -g_k
            
        elif metodo == "Método de Newton":
            H_k = calcular_hessiana(f_sym, vars_sym, x_k)
            try:
                # Resolver H_k * d_k = -g_k
                d_k = np.linalg.solve(H_k, -g_k)
            except np.linalg.LinAlgError:
                # Si la Hessiana no es invertible, usamos el gradiente como respaldo
                d_k = -g_k
                
        elif metodo == "Gradiente Conjugado":
            if i == 0 or d_ant is None:
                d_k = -g_k
            else:
                # Fórmula de Fletcher-Reeves
                beta = np.dot(g_k, g_k) / max(np.dot(g_ant, g_ant), 1e-10)
                d_k = -g_k + beta * d_ant
                # Reinicio si deja de ser dirección de descenso
                if np.dot(d_k, g_k) > 0:
                    d_k = -g_k
            d_ant = d_k.copy()
            g_ant = g_k.copy()
            
        # Encontrar paso alpha mediante Condiciones de Wolfe
        alpha = busqueda_linea_wolfe(evaluar_funcion, calcular_gradiente, f_sym, vars_sym, x_k, d_k, c1, c2)
        
        # Actualizar punto
        x_k = x_k + alpha * d_k
        historial_puntos.append(x_k.copy())
        
    return x_k, historial_puntos, i + 1, norma_g

# --- INTERFAZ DE USUARIO ---

col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.header("📥 Datos de Entrada")
    
    num_vars = st.selectbox("Número de variables", [1, 2])
    metodo = st.selectbox("Método de optimización", ["Método del Gradiente", "Gradiente Conjugado", "Método de Newton"])
    
    # Sugerencias dinámicas según variables
    if num_vars == 1:
        funcion_sug = "x**2 + 4*x + 4"
    else:
        funcion_sug = "x**2 + y**2 - 2*x - 4*y"
        
    func_input = st.text_input("Función objetivo (en formato Python)", value=funcion_sug)
    
    st.subheader("Punto de partida")
    if num_vars == 1:
        x0_val = st.number_input("x0", value=2.0)
        punto_partida = [x0_val]
    else:
        x0_val = st.number_input("x0", value=4.0)
        y0_val = st.number_input("y0", value=4.0)
        punto_partida = [x0_val, y0_val]
        
    st.subheader("Parámetros de Control")
    max_iter = st.number_input("Número máximo de iteraciones", value=100, step=10)
    tol = st.number_input("Tolerancia de convergencia", value=1e-5, format="%.5f")
    
    st.subheader("Condiciones de Wolfe")
    c1 = st.number_input("Parámetro c1 (Armijo)", value=1e-4, format="%.4f")
    c2 = st.number_input("Parámetro c2 (Curvatura)", value=0.9, format="%.2f")

with col_der:
    st.header("📊 Resultados del Algoritmo")
    
    if st.button("▶️ Ejecutar Optimización", type="primary"):
        try:
            # Parsear la función usando SymPy
            if num_vars == 1:
                x = sp.Symbol('x')
                vars_sym = [x]
            else:
                x, y = sp.symbols('x y')
                vars_sym = [x, y]
                
            f_sym = sp.sympify(func_input)
            
            # Ejecutar el algoritmo matemático
            min_encontrado, historial, iters, error_f = optimizar(
                metodo, f_sym, vars_sym, punto_partida, max_iter, tol, c1, c2
            )
            
            valor_final_f = evaluar_funcion(f_sym, vars_sym, min_encontrado)
            
            # Mostrar los resultados exactos calculados en los componentes de la interfaz
            if num_vars == 1:
                texto_minimo = f"({min_encontrado[0]:.4f})"
            else:
                texto_minimo = f"({min_encontrado[0]:.4f} ; {min_encontrado[1]:.4f})"
                
            st.success(f"🎯 Punto mínimo encontrado: **{texto_minimo}**")
            
            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric(label="Valor de f(x*)", value=f"{valor_final_f:.5f}")
            c_m2.metric(label="Iteraciones realizadas", value=str(iters))
            c_m3.metric(label="Error final (||∇f||)", value=f"{error_f:.5e}")
            
            # --- GENERACIÓN DEL GRÁFICO (CURVAS DE NIVEL) ---
            st.subheader("📈 Gráfico de Contorno y Camino de Convergencia")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            historial = np.array(historial)
            
            if num_vars == 2:
                # Crear malla de puntos para las curvas de nivel basándose en el recorrido
                x_min, x_max = min(historial[:, 0].min() - 1, punto_partida[0] - 1), max(historial[:, 0].max() + 1, punto_partida[0] + 1)
                y_min, y_max = min(historial[:, 1].min() - 1, punto_partida[1] - 1), max(historial[:, 1].max() + 1, punto_partida[1] + 1)
                
                X, Y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                
                # Evaluar la función en toda la malla
                Z = np.zeros_like(X)
                for i_m in range(X.shape[0]):
                    for j_m in range(X.shape[1]):
                        Z[i_m, j_m] = evaluar_funcion(f_sym, vars_sym, [X[i_m, j_m], Y[i_m, j_m]])
                
                # Dibujar las curvas de nivel del gráfico
                cp = ax.contour(X, Y, Z, levels=20, cmap='viridis')
                ax.clabel(cp, inline=True, fontSize=8)
                
                # Graficar el camino que hizo el algoritmo (puntos rojos unidos por líneas)
                ax.plot(historial[:, 0], historial[:, 1], 'r.-', label="Camino de optimización", markersize=8)
                ax.plot(punto_partida[0], punto_partida[1], 'go', label="Inicio")
                ax.plot(min_encontrado[0], min_encontrado[1], 'b*', label="Mínimo", markersize=12)
                
                ax.set_xlabel('Variable X')
                ax.set_ylabel('Variable Y')
                ax.legend()
                ax.grid(True)
                
            else:
                # Gráfico 2D simple para 1 sola variable
                x_min, x_max = historial[:, 0].min() - 2, historial[:, 0].max() + 2
                X = np.linspace(x_min, x_max, 200)
                Y = [evaluar_funcion(f_sym, vars_sym, [val]) for val in X]
                
                ax.plot(X, Y, 'b-', label='f(x)')
                Y_hist = [evaluar_funcion(f_sym, vars_sym, [val]) for val in historial[:, 0]]
                ax.plot(historial[:, 0], Y_hist, 'r.-', label='Iteraciones')
                ax.set_xlabel('Variable X')
                ax.set_ylabel('f(x)')
                ax.legend()
                ax.grid(True)
                
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"⚠️ Hubo un error en la lectura de la función o algoritmo. Verifica el formato. Error: {e}")
