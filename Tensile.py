import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def read_lammps_output(filename):
    """Membaca data strain dan stress dari file output LAMMPS."""
    strain = []
    stress = []
    
    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) >= 2:  # Pastikan ada cukup data di setiap baris
                try:
                    strain.append(float(data[0]))  # Asumsikan kolom pertama adalah strain
                    stress.append(float(data[1]))  # Asumsikan kolom kedua adalah stress
                except ValueError:
                    continue  # Lewati jika ada kesalahan konversi
    
    return np.array(strain), np.array(stress)

def detect_linear_region(strain, stress):
    """
    Mendeteksi daerah linier dari kurva stress-strain menggunakan regresi linear.
    Young's Modulus dihitung sebagai gradien dari bagian awal kurva hingga UTS.
    """
    max_stress = np.max(stress)  # Ambil UTS
    uts_index = np.argmax(stress)  # Ambil indeks UTS
    
    # Gunakan regresi linear dari data awal hingga titik UTS
    slope, intercept, r_value, _, _ = linregress(strain[:uts_index], stress[:uts_index])
    
    return slope, intercept, r_value**2  # Mengembalikan gradien, intercept, dan R^2

def plot_stress_strain(strain, stress, elastic_modulus, intercept, r_squared, uts, output_filename):
    """Membuat dan menyimpan grafik stress-strain dengan garis regresi Young's Modulus diperpanjang."""
    plt.figure(figsize=(8, 6))
    plt.plot(strain, stress, marker='o', linestyle='-', color='b', label='Stress-Strain Curve')

    # Perpanjang garis Young's Modulus hingga mencapai UTS
    max_stress = np.max(stress)
    max_strain = (max_stress - intercept) / elastic_modulus  # Perpanjangan strain hingga UTS
    strain_extended = np.linspace(0, max_strain, 100)  
    stress_extended = elastic_modulus * strain_extended + intercept  

    plt.plot(strain_extended, stress_extended, linestyle='--', color='r', label="Young's Modulus Fit")

    # Tambahkan teks persamaan garis dan nilai R^2 di grafik
    equation_text = f"y = {elastic_modulus:.3f}x + {intercept:.3f}\nR² = {r_squared:.3f}"
    plt.text(0.6 * max_strain, 0.8 * max_stress, equation_text, fontsize=10, color='red',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    # Tambahkan teks UTS di grafik
    plt.axhline(y=uts, color='g', linestyle='--', label=f'UTS = {uts:.3f} GPa')
    plt.text(0.05, uts * 1.05, f'UTS = {uts:.3f} GPa', fontsize=10, color='green',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))

    plt.xlabel('Strain')
    plt.ylabel('Stress (GPa)')
    plt.title('Grafik Stress-Strain Mg1%Ca')
    plt.legend()
    plt.grid()

    # Atur batas sumbu y agar tidak memotong data negatif
    min_stress = np.min(stress)
    plt.ylim(min_stress * 1.1, max_stress * 1.2)  # Biarkan nilai negatif tampil

    plt.savefig(output_filename, dpi=300)  # Simpan gambar sebagai file PNG
    plt.show()

def calculate_uts(stress):
    """Menghitung Ultimate Tensile Strength (UTS) dari data stress."""
    return np.max(stress)  # UTS adalah nilai maksimum dari stress

# Ganti dengan nama file output yang sesuai
filename = "CaMg_stress_strain.txt"
output_image = "stress_strain_curve.png"
strain, stress = read_lammps_output(filename)

# Hitung UTS
uts = calculate_uts(stress)
print(f"Ultimate Tensile Strength (UTS): {uts:.3f} GPa")

# Deteksi dan hitung Modulus Elastisitas (Young's Modulus) di daerah linier
elastic_modulus, intercept, r_squared = detect_linear_region(strain, stress)
print(f"Elastic Modulus (Young's Modulus): {elastic_modulus:.3f} GPa")
print(f"Coefficient of Determination (R²): {r_squared:.3f}")

# Tampilkan grafik dengan teks persamaan garis, R², dan nilai UTS
plot_stress_strain(strain, stress, elastic_modulus, intercept, r_squared, uts, output_image)
