# 🧠 Domain AI Expert Report: Fundamentos Físicos del Diagnóstico de Rodamientos por Vibraciones

## 1. Principio de Operación

Los rodamientos de elementos rodantes son componentes críticos en la maquinaria rotativa. Cuando presentan defectos en sus superficies de contacto (pistas interna y externa, elementos rodantes o jaula), cada paso de un elemento rodante sobre el defecto genera un impacto mecánico breve de gran energía.

Estos impactos excitan las frecuencias naturales de la estructura circundante y del sensor (acelerómetro). La periodicidad característica con la que ocurren estos impactos produce un patrón espectral modulado. Analizando las frecuencias de paso específicas en el espectro de la señal o en su envolvente, es posible identificar con exactitud tanto la existencia como la localización del fallo mecánico antes de que este resulte en una parada de planta catastrófica.

---

## 2. Frecuencias de Falla Teóricas

Las frecuencias características de falla de un rodamiento se calculan analíticamente a partir de su geometría constructiva y de la velocidad de rotación del eje motor.

| Frecuencia | Símbolo | Descripción |
| :--- | :--- | :--- |
| **BPFO** | Ball Pass Frequency Outer Race | Frecuencia de paso de bolas por la pista externa |
| **BPFI** | Ball Pass Frequency Inner Race | Frecuencia de paso de bolas por la pista interna |
| **BSF** | Ball Spin Frequency | Frecuencia de giro del elemento rodante |
| **FTF** | Fundamental Train Frequency | Frecuencia de rotación de la jaula porta-bolas |

### Ecuaciones Cinemáticas de Falla:

$$BPFO = \frac{N_b}{2} \cdot f_r \cdot \left(1 - \frac{B_d}{P_d} \cos\phi\right)$$

$$BPFI = \frac{N_b}{2} \cdot f_r \cdot \left(1 + \frac{B_d}{P_d} \cos\phi\right)$$

$$BSF = \frac{P_d}{2 \cdot B_d} \cdot f_r \cdot \left(1 - \left(\frac{B_d}{P_d} \cos\phi\right)^2\right)$$

$$FTF = \frac{f_r}{2} \cdot \left(1 - \frac{B_d}{P_d} \cos\phi\right)$$

#### Donde:
* $N_b$ = Número de elementos rodantes (bolas/rodillos)
* $f_r$ = Frecuencia de rotación del eje [Hz] (RPM / 60)
* $B_d$ = Diámetro de la bola o elemento rodante [mm]
* $P_d$ = Diámetro primitivo del rodamiento (Pitch Diameter) [mm]
* $\phi$ = Ángulo de contacto de los elementos rodantes con las pistas [grados]

---

## 3. Procesamiento Digital de Señal (DSP)

Para extraer indicadores robustos de una señal con alto ruido de fondo (ruido electromagnético de motores, fricción estándar, etc.), se implementa una cadena secuencial de procesamiento analítico:

```
[Señal de Vibración Cruda (Acelerómetro)]
                  │
                  ▼
       [Supresión de DC Offset]  <─── Resta de la media estadística
                  │
                  ▼
         [Indicador RMS Global]  <─── Cálculo de energía total acumulada
                  │
                  ▼
      [FFT & Análisis Espectral] <─── Resolución espectral de 1.22 Hz
                  │
                  ▼
     [Demodulación por Envolvente] <─ Transformada de Hilbert para fallas incipientes
```

### 3.1 Supresión de DC Offset
La componente de corriente continua (DC offset) presente en los acelerómetros comerciales genera una distorsión crítica en el espectro tras aplicar la Transformada Rápida de Fourier (FFT), manifestándose como un pico artificial masivo en $0\text{ Hz}$ que enmascara las bajas frecuencias (como la FTF de la jaula).

El sistema robustece el análisis centrando la señal a media cero antes de cualquier transformación espectral:

$$\bar{x}[n] = x[n] - \frac{1}{N}\sum_{i=1}^{N} x[i]$$

### 3.2 RMS (Root Mean Square)
El valor eficaz RMS representa la energía total de la vibración y es el indicador primario estándar de la severidad del daño (ISO 10816). Su cálculo matemático se define como:

$$RMS = \sqrt{\frac{1}{N}\sum_{n=1}^{N} \bar{x}[n]^2}$$

### 3.3 FFT (Fast Fourier Transform)
Permite descomponer la señal temporal en sus componentes de frecuencia elementales. La resolución espectral ($\Delta f$) determina el nivel de detalle y discriminación entre picos de frecuencia muy cercanos:

$$\Delta f = \frac{F_s}{N} = \frac{20000\text{ Hz}}{16384} \approx 1.22\text{ Hz}$$

Esta resolución permite diferenciar con total precisión frecuencias vecinas como la rotación del eje ($f_r \approx 33.3\text{ Hz}$ a $2000\text{ RPM}$) y la jaula ($FTF \approx 14.8\text{ Hz}$).

### 3.4 Demodulación por Envolvente
Los defectos incipientes (microfisuras iniciales) generan pulsos de impacto de muy corta duración que excitan las frecuencias de resonancia del sistema a alta frecuencia (típicamente entre $2\text{ kHz}$ y $10\text{ kHz}$). Debido a la distancia, estas amplitudes son muy débiles y quedan ocultas por la vibración general de baja frecuencia de la máquina rotativa.

La **Transformada de Hilbert** se aplica para extraer la envolvente de la señal de alta frecuencia filtrada por paso banda, permitiendo recuperar la baja frecuencia de repetición de los impactos (modulación), permitiendo diagnosticar la falla antes de que la energía RMS global se vea incrementada en términos absolutos.

---

## 4. Modelo de Degradación y Pronóstico (RUL)

### 4.1 Comportamiento en Tres Fases
La vida útil de un rodamiento bajo fatiga y carga constante no es lineal. Sigue una dinámica bien definida de tres etapas fundamentales:

1. **Fase I (Saludable):** Operación en régimen normal con desgaste microscópico. El RMS es extremadamente bajo y estable, dominado únicamente por el ruido base de la máquina.
2. **Fase II (Degradación):** Inicio de la propagación del defecto mecánico superficial (microcrack / desprendimiento). El RMS experimenta un incremento de carácter exponencial conforme el tamaño de la picadura aumenta de forma auto-acelerada.
3. **Fase III (Crítico):** Severidad extrema con pérdida del comportamiento suave. La fricción y los atascos momentáneos causan lecturas RMS elevadas con fluctuaciones erráticas severas previas al fallo mecánico catastrófico de rotura.

```
RMS (g)
  ▲
  │                                                     * Falla Catastrófica (Fase III)
  │                                                   * *
  │                                                 *
  │                                               *  <─── Inicio de Fase II (Inflexión Log)
  │                                            *
  │  ───────────────────────────────────────*──────────── Umbral Crítico (0.25 g)
  │                                     *
  │  ────────────────────────────────*─────────────────── Umbral de Alerta (0.12 g)
  │                              *
  │___________________________*
  │  Fase I (Saludable)      │    Fase II (Degradación)
  └──────────────────────────┴──────────────────────────► Tiempo / Ciclos
```

### 4.2 Regresión Híbrida y Log-Exponencial
Para estimar con precisión el tiempo de vida útil restante (**RUL - Remaining Useful Life**), el sistema emplea un **Motor Híbrido** que evalúa continuamente dos modelos sobre la ventana activa de mediciones temporal:
* **Modelo Lineal:** Ajusta una recta de regresión directa en el dominio físico. Excelente para fases estables o crecimientos lentos y controlados.
* **Modelo Exponencial (Log-Lineal):** Se aplica una regresión lineal sobre el logaritmo natural de la variable RMS:

  $$\ln(RMS) = m \cdot (x - x_0) + b$$

  Esto modela un crecimiento puramente exponencial en la variable real $RMS$:

  $$RMS(x) = e^b \cdot e^{m \cdot (x - x_0)}$$

Para evitar el desbordamiento numérico (overflow) en cálculos exponenciales rápidos sobre ciclos de gran tamaño, el origen temporal de la regresión se desplaza dinámicamente al instante actual de evaluación ($x_0$). El sistema selecciona automáticamente el modelo con menor error cuadrático medio (RMSE) para realizar la proyección.

### 4.3 Válvula Check de Seguridad (Daño Físicamente Irreversible)
Un defecto mecánico en un rodamiento es un fenómeno de desgaste destructivo físicamente irreversible. Sin embargo, debido a fenómenos de autolubricación por desprendimiento de grasa, o al desgaste abrasivo de los bordes agudos de una grieta, el valor eficaz del RMS puede experimentar caídas temporales de forma transitoria.

Para evitar falsos diagnósticos que pongan en peligro la seguridad industrial, el sistema implementa una **válvula check de seguridad**. Esta lógica bloquea la reducción del estado crítico mediante el mantenimiento estricto de la variable `max_rms_historico`. El estado de alerta del rodamiento solo puede incrementarse y jamás recupera un nivel inferior estable de salud sin registrarse un cambio físico de activo.

### 4.4 Discriminación de Eje Detenido (Filtro Antiruido)
Si la maquinaria es detenida por producción, mantenimiento o fin de turno, el acelerómetro sigue leyendo ruido eléctrico instrumental de fondo. Si el modelo intentase calcular el RUL sobre esta caída drástica de RMS, generaría falsos positivos de recuperación o errores matemáticos severos.

El sistema implementa una regla lógica de corte: si el RMS cae por debajo del ruido base del acelerómetro calibrado (`motor_apagado_rms = 0.01 g`), el estado se clasifica de inmediato como **Fuera de Servicio (Eje Detenido)**, bloqueando todos los algoritmos de proyección predictiva de RUL para conservar la integridad del histórico analítico.

---

## 5. Dataset de Referencia

El sistema opera sobre los históricos reales del **NASA IMS Bearing Dataset**, un hito internacional en el área de la analítica prognóstica compilado por el *Center for Intelligent Maintenance Systems* (Universidad de Cincinnati) bajo convenio de la NASA.

* **Estructura:** Tres ensayos acelerados de vida bajo carga radial constante de $6000\text{ lbs}$.
* **Frecuencia de Muestreo ($F_s$):** $20,000\text{ Hz}$ por canal de lectura.
* **Estructura del Burst:** Captura de $16,384$ muestras crudas cada $10\text{ minutos}$.
* **Almacenamiento Consolidado:** Los datos crudos en formato plano son transformados a través de nuestro pipeline eficiente en estructuras columnares eficientes de formato **Parquet, CSV** y base de datos relacional de indexación indexada en **SQLite** para soportar el análisis multi-activo en tiempo real de manera escalable.

---

## 6. Referencias Bibliográficas

* **Randall, R. B.** (2011). *Vibration-based Condition Monitoring: Industrial, Aerospace and Automotive Applications*. Wiley. ISBN: 978-0470747759.
* **Qiu, H., Lee, J., Lin, J., & Yu, G.** (2006). *Wavelet filter-based weak signature detection with its application on rolling element bearing prognosis*. *Journal of Sound and Vibration*, 289(4-5), 1066-1090.
* **NASA Prognostics Data Repository.** *IMS Bearing Dataset*. Center for Intelligent Maintenance Systems, University of Cincinnati.
