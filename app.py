# app.py
# Flask backend – robust, Qiskit-2.0-compatible Hamiltonian construction,
# multi-material composite handling, mock/safe optimization & circuit image generation.

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64, math, json
from qiskit.quantum_info import SparsePauliOp
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime

app = Flask(__name__)

# --- Extended Material library (150+ nanomaterials) ---
MATERIAL_LIBRARY = [
    # Metal Oxides (50)
    "TiO2", "ZnO", "CdS", "WO3", "BiVO4", "Fe2O3", "Cu2O", "SnO2", "NiO", "Co3O4",
    "In2O3", "SrTiO3", "BaTiO3", "Fe3O4", "CuO", "CeO2", "LaFeO3", "V2O5", "MnO2",
    "Nb2O5", "ZrO2", "Al2O3", "Bi2WO6", "Bi2O3", "Ta2O5", "RuO2", "IrO2", "Y2O3",
    "MgO", "Ga2O3", "GeO2", "PbO", "VO2", "Cr2O3", "NiCo2O4", "ZnFe2O4",
    "CoFe2O4", "MnFe2O4", "CuFe2O4", "NiFe2O4", "Mn3O4", "Fe3O4-NPs", "TiO2-P25",
    "ZnO-NRs", "WO3-NPs", "SnO2-QDs", "CuO-NWs", "NiO-NPs", "TiO2-NT", "ZnO-NWs",
    
    # Sulfides & Selenides (25)
    "CdSe", "ZnS", "Cu2S", "Sb2S3", "PbS", "SnS2", "Bi2S3", "MoS2", "WS2", "CuInS2",
    "ZnIn2S4", "CdIn2S4", "Ag2S", "CuS", "FeS2", "In2S3", "CdTe", "PbSe", "Bi2Se3",
    "MoSe2", "WSe2", "SnSe", "Sb2Se3", "Cu2Se", "Ag2Se",
    
    # Perovskites (15)
    "CsPbBr3", "CsPbI3", "CsPbCl3", "MAPbI3", "FAPbI3", "CsSnI3", "LaCoO3", "LaNiO3",
    "SrTiO3-NPs", "BaTiO3-NPs", "CaTiO3", "PbTiO3", "BiFeO3", "LaAlO3", "NdGaO3",
    
    # Carbon-based (20)
    "gC3N4", "C3N4x", "BlackP", "Graphene", "CNT", "Fullerene", "GrapheneOxide",
    "ReducedGO", "CarbonDots", "Graphdiyne", "CNT-MWCNT", "CNT-SWCNT", "C60", "C70",
    "GrapheneNR", "N-Graphene", "B-Graphene", "S-Graphene", "P-Graphene", "Graphyne",
    
    # Noble Metal Decorated (15)
    "Pt/TiO2", "Au/TiO2", "Pd/CuO", "Ag/TiO2", "Au/ZnO", "Pt/gC3N4", "Pd/BiVO4",
    "Ru/TiO2", "Rh/TiO2", "Ag/BiVO4", "Au/CdS", "Pt/WO3", "Pd/ZnO", "Ag/gC3N4", "Au/MoS2",
    
    # Nitrides & Phosphides (15)
    "TiN", "NbN", "GaN", "InN", "AlN", "Ta3N5", "Ni2P", "CoP", "FeP", "Cu3P",
    "MoN", "WN", "VN", "CrN", "ZrN",
    
    # MOFs & Hybrid Materials (10)
    "UiO66", "MIL101", "ZIF8", "HKUST1", "MOF5", "PCN222", "MXeneTi3C2", "MXeneV2C",
    "COF-1", "ZIF-67",
    
    # Advanced & Mixed Materials (20)
    "BiOCl", "BiOBr", "BiOI", "WO3-x", "MoOx", "ReO3", "Ag3PO4", "Ag2O", "CuWO4",
    "CuBi2O4", "ZnWO4", "AgBr", "AgCl", "AgI", "BiPO4", "Bi4Ti3O12", "CaBi2O4",
    "ZnV2O6", "BiVO4-NPs", "Fe2TiO5"
]

# Extended base bandgaps
BASE_BANDGAP = {
    "TiO2": 3.2, "ZnO": 3.3, "CdS": 2.4, "WO3": 2.6, "BiVO4": 2.4, "Fe2O3": 2.2,
    "gC3N4": 2.7, "Cu2O": 2.1, "MoS2": 1.9, "SnO2": 3.6, "Ag3PO4": 2.5, "NiO": 3.5,
    "Co3O4": 1.8, "In2O3": 2.9, "SrTiO3": 3.1, "ZnIn2S4": 2.2, "CdSe": 1.7,
    "WS2": 2.0, "GaN": 3.4, "Ta3N5": 2.1, "CsPbBr3": 2.3, "BlackP": 1.5,
    "Graphene": 0.0, "BaTiO3": 3.2, "LaFeO3": 2.1, "V2O5": 2.3, "MnO2": 2.5,
    "CuO": 1.4, "SnS2": 2.2, "Bi2S3": 1.3, "CdTe": 1.5, "PbSe": 0.3
}
DEFAULT_BANDGAP = 2.5

# Map circuit depth selection to qubit counts
DEPTH_TO_QUBITS = {
    2: 6,
    3: 10,
    4: 15,
    5: 20
}

def create_hamiltonian(num_qubits, params):
    """Build SparsePauliOp Hamiltonian"""
    materials = params.get("materials", ["TiO2"])
    bandgaps = [BASE_BANDGAP.get(m, DEFAULT_BANDGAP) for m in materials]
    avg_bandgap = sum(bandgaps) / max(1, len(bandgaps))
    params["targetBandgap"] = avg_bandgap

    pauli_terms = []

    for i in range(num_qubits):
        label = ''.join(['Z' if j == i else 'I' for j in range(num_qubits)])
        coeff = -avg_bandgap * 0.5
        pauli_terms.append((label, coeff))

    surf = float(params.get("surfaceArea", 100.0))
    coupling_scale = -surf * 0.01
    for i in range(num_qubits - 1):
        label = ''.join(['Z' if j in (i, i+1) else 'I' for j in range(num_qubits)])
        pauli_terms.append((label, coupling_scale))

    psize = float(params.get("particleSize", 50.0))
    x_scale = -psize * 0.005
    for i in range(num_qubits):
        label = ''.join(['X' if j == i else 'I' for j in range(num_qubits)])
        pauli_terms.append((label, x_scale))

    spop = SparsePauliOp.from_list(pauli_terms)
    return spop

def simple_classical_optimization(hamiltonian: SparsePauliOp, ansatz_qubits: int, max_iters=120):
    """Classical optimization simulator"""
    n_params = max(6, min(ansatz_qubits * 2, 50))
    rng = np.random.default_rng(seed=42)
    best_theta = rng.uniform(0, 2*np.pi, n_params)
    coeffs = hamiltonian.coeffs
    scale = float(np.sum(np.abs(coeffs))) / max(1.0, len(coeffs))

    def cost(theta):
        s = np.sum(np.cos(theta) * np.sin(theta*0.5))
        return scale * (1.0 + 0.5 * np.tanh(s))

    best_val = cost(best_theta)
    history = [best_val]
    for it in range(max_iters):
        cand = best_theta + rng.normal(0, 0.5, size=n_params) * (0.9 ** (it/10))
        val = cost(cand)
        if val < best_val:
            best_val, best_theta = val, cand
        history.append(best_val)

    return {
        "energy": float(best_val),
        "theta": best_theta.tolist(),
        "history": [float(x) for x in history]
    }

def generate_circuit_image_placeholder(num_qubits, n_gates=12):
    """Generate enhanced circuit visualization with better graphics"""
    fig, ax = plt.subplots(figsize=(10, max(3, num_qubits*0.35)))
    
    # Dark background with gradient effect
    ax.set_facecolor('#0a0e1a')
    fig.patch.set_facecolor('#060a14')
    ax.axis('off')
    
    # Calculate qubit positions
    ys = np.linspace(0.92, 0.08, num_qubits)
    
    # Draw qubit lines with glow effect
    for i, y in enumerate(ys):
        # Main qubit line
        ax.hlines(y, 0.05, 0.95, color='#2a3f5f', linewidth=2.5, alpha=0.9, zorder=1)
        # Glow effect
        ax.hlines(y, 0.05, 0.95, color='#00d8ff', linewidth=4, alpha=0.15, zorder=0)
        
        # Qubit label with enhanced styling
        ax.text(0.02, y, f"q[{i}]", color='#7fd8ff', fontsize=9, 
               va='center', family='monospace', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f1829', 
                        edgecolor='#00d8ff', alpha=0.8, linewidth=1))
    
    # Gate types with different colors
    gate_types = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CNOT']
    gate_colors = {
        'H': '#00d8ff',   # Cyan
        'X': '#ff00d4',   # Magenta
        'Y': '#00ff88',   # Green
        'Z': '#ffaa00',   # Orange
        'RX': '#ff5555',  # Red
        'RY': '#aa55ff',  # Purple
        'RZ': '#55aaff',  # Blue
        'CNOT': '#ffdd00' # Yellow
    }
    
    rng = np.random.default_rng(123)
    
    # Increase number of gates for better visual
    n_gates = min(num_qubits * 2, 20)
    
    # Draw quantum gates with enhanced styling
    for g in range(n_gates):
        gx = 0.12 + (g / n_gates) * 0.78
        row = rng.integers(0, num_qubits)
        y = ys[row]
        
        # Select gate type
        gate_type = rng.choice(['H', 'RX', 'RY', 'RZ', 'X', 'Y', 'Z'])
        gate_color = gate_colors[gate_type]
        
        # Draw gate shadow for depth
        shadow = plt.Rectangle((gx-0.024, y-0.024), 0.048, 0.048, 
                              color='#000000', alpha=0.5, zorder=2)
        ax.add_patch(shadow)
        
        # Draw main gate box with gradient effect
        rect = plt.Rectangle((gx-0.025, y-0.025), 0.05, 0.05, 
                            color=gate_color, ec='#ffffff', 
                            alpha=0.9, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        
        # Add inner glow
        glow = plt.Rectangle((gx-0.025, y-0.025), 0.05, 0.05, 
                            color=gate_color, alpha=0.3, linewidth=0, zorder=2.5)
        ax.add_patch(glow)
        
        # Gate label
        ax.text(gx, y, gate_type, color='#0a0e1a', ha='center', 
               va='center', fontsize=7, weight='bold', zorder=4,
               family='monospace')
    
    # Add some CNOT gates (two-qubit gates)
    n_cnot = min(3, num_qubits // 4)
    for c in range(n_cnot):
        gx = 0.2 + (c / max(1, n_cnot-1)) * 0.6 if n_cnot > 1 else 0.5
        control = rng.integers(0, max(1, num_qubits-1))
        target = rng.integers(control+1, num_qubits)
        
        y_control = ys[control]
        y_target = ys[target]
        
        # Draw CNOT line
        ax.plot([gx, gx], [y_control, y_target], 
               color='#ffdd00', linewidth=2.5, alpha=0.8, zorder=3)
        
        # Control dot
        ax.scatter([gx], [y_control], s=80, color='#ffdd00', 
                  edgecolors='#ffffff', linewidth=1.5, zorder=4)
        
        # Target circle (⊕)
        target_circle = plt.Circle((gx, y_target), 0.02, 
                                  color='none', ec='#ffdd00', linewidth=2.5, zorder=4)
        ax.add_patch(target_circle)
        
        # Target cross
        cross_size = 0.015
        ax.plot([gx-cross_size, gx+cross_size], [y_target, y_target], 
               color='#ffdd00', linewidth=2.5, zorder=4)
        ax.plot([gx, gx], [y_target-cross_size, y_target+cross_size], 
               color='#ffdd00', linewidth=2.5, zorder=4)
    
    # Title with enhanced styling
    title_text = f"Quantum Circuit - {num_qubits} Qubits"
    ax.text(0.5, 0.98, title_text, color='#ffffff', fontsize=13, 
           ha='center', weight='bold', family='sans-serif',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a2744', 
                    edgecolor='#00d8ff', alpha=0.9, linewidth=2))
    
    # Add decorative corners
    corner_size = 0.02
    # Top-left
    ax.plot([0.03, 0.03+corner_size], [0.97, 0.97], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    ax.plot([0.03, 0.03], [0.97, 0.97-corner_size], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    # Top-right
    ax.plot([0.97-corner_size, 0.97], [0.97, 0.97], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    ax.plot([0.97, 0.97], [0.97, 0.97-corner_size], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    # Bottom-left
    ax.plot([0.03, 0.03+corner_size], [0.03, 0.03], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    ax.plot([0.03, 0.03], [0.03, 0.03+corner_size], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    # Bottom-right
    ax.plot([0.97-corner_size, 0.97], [0.03, 0.03], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    ax.plot([0.97, 0.97], [0.03, 0.03+corner_size], 
           color='#00d8ff', linewidth=2, alpha=0.6)
    
    # Add measurement symbols at the end
    for i, y in enumerate(ys):
        # Measurement box
        mx = 0.96
        mbox = plt.Rectangle((mx-0.015, y-0.015), 0.03, 0.03,
                            color='#2a4f7f', ec='#00d8ff',
                            alpha=0.9, linewidth=1.5, zorder=3)
        ax.add_patch(mbox)
        
        # Measurement arc (simplified meter symbol)
        ax.text(mx, y, 'M', color='#00d8ff', ha='center', 
               va='center', fontsize=6, weight='bold', zorder=4)
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', 
               facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

@app.route('/')
def index():
    return render_template('index.html', materials=MATERIAL_LIBRARY)

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        params = request.get_json(force=True) or {}
        materials = params.get('materials') or params.get('materialType') or ["TiO2"]
        if isinstance(materials, str):
            materials = [m.strip() for m in materials.split(',') if m.strip()]
        if not materials:
            materials = ["TiO2"]
        params['materials'] = materials

        depth = int(params.get('circuitDepth', 4))
        num_qubits = int(params.get('qubits')) if params.get('qubits') else DEPTH_TO_QUBITS.get(depth, 10)

        hamiltonian = create_hamiltonian(num_qubits, params)
        opt_res = simple_classical_optimization(hamiltonian, ansatz_qubits=num_qubits, max_iters=80)
        circuit_img = generate_circuit_image_placeholder(num_qubits)

        energy = opt_res["energy"]
        bandgap = params.get('targetBandgap', DEFAULT_BANDGAP)
        efficiency = min(99.0, max(5.0, 60.0 + (1.0 / (1.0 + abs(energy))) * 40.0))
        absorption = min(100.0, 40.0 + (bandgap / 4.0) * 60.0)
        quantum_yield = efficiency * 0.82
        stability = min(100.0, 85.0 - abs(energy)*5.0)

        result = {
            "success": True,
            "materials": materials,
            "num_qubits": num_qubits,
            "circuit_depth": depth,
            "final_energy": float(energy),
            "surface_area": float(params.get("surfaceArea", 100.0)),
            "particle_size": float(params.get("particleSize", 50.0)),
            "optimized_params": {
                "bandgap": float(bandgap),
                "efficiency": float(efficiency),
                "absorption": float(absorption),
                "quantum_yield": float(quantum_yield),
                "stability": float(stability)
            },
            "history": opt_res["history"],
            "circuit_image": circuit_img,
            "analysis": f"Composite materials: {', '.join(materials)} – averaged bandgap {bandgap:.2f} eV. Optimization used {num_qubits} qubits (simulated)."
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download/excel', methods=['POST'])
def download_excel():
    """Generate and download Excel report"""
    try:
        data = request.get_json(force=True)
        
        wb = Workbook()
        
        # Summary Sheet
        ws1 = wb.active
        ws1.title = "Summary"
        
        # Header styling
        header_fill = PatternFill(start_color="00D8FF", end_color="00D8FF", fill_type="solid")
        header_font = Font(bold=True, size=12, color="000000")
        
        ws1['A1'] = "QUANTUM PHOTOCATALYST OPTIMIZATION REPORT"
        ws1['A1'].font = Font(bold=True, size=16, color="0066CC")
        ws1.merge_cells('A1:D1')
        
        ws1['A3'] = "Generated:"
        ws1['B3'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        ws1['A5'] = "MATERIAL CONFIGURATION"
        ws1['A5'].font = header_font
        ws1['A5'].fill = header_fill
        ws1.merge_cells('A5:D5')
        
        ws1['A6'] = "Selected Materials:"
        ws1['B6'] = ", ".join(data.get('materials', []))
        ws1.merge_cells('B6:D6')
        
        ws1['A7'] = "Number of Materials:"
        ws1['B7'] = len(data.get('materials', []))
        
        ws1['A9'] = "QUANTUM CIRCUIT PARAMETERS"
        ws1['A9'].font = header_font
        ws1['A9'].fill = header_fill
        ws1.merge_cells('A9:D9')
        
        ws1['A10'] = "Number of Qubits:"
        ws1['B10'] = data.get('num_qubits', 0)
        
        ws1['A11'] = "Circuit Depth Level:"
        ws1['B11'] = data.get('circuit_depth', 0)
        
        ws1['A12'] = "Final Energy:"
        ws1['B12'] = round(data.get('final_energy', 0), 6)
        
        ws1['A14'] = "MATERIAL PROPERTIES"
        ws1['A14'].font = header_font
        ws1['A14'].fill = header_fill
        ws1.merge_cells('A14:D14')
        
        ws1['A15'] = "Surface Area (m²/g):"
        ws1['B15'] = data.get('surface_area', 0)
        
        ws1['A16'] = "Particle Size (nm):"
        ws1['B16'] = data.get('particle_size', 0)
        
        ws1['A18'] = "PERFORMANCE METRICS"
        ws1['A18'].font = header_font
        ws1['A18'].fill = header_fill
        ws1.merge_cells('A18:D18')
        
        params = data.get('optimized_params', {})
        ws1['A19'] = "Bandgap (eV):"
        ws1['B19'] = round(params.get('bandgap', 0), 2)
        
        ws1['A20'] = "Efficiency (%):"
        ws1['B20'] = round(params.get('efficiency', 0), 2)
        
        ws1['A21'] = "Absorption (%):"
        ws1['B21'] = round(params.get('absorption', 0), 2)
        
        ws1['A22'] = "Quantum Yield (%):"
        ws1['B22'] = round(params.get('quantum_yield', 0), 2)
        
        ws1['A23'] = "Stability (%):"
        ws1['B23'] = round(params.get('stability', 0), 2)
        
        # Materials List Sheet
        ws2 = wb.create_sheet("Materials List")
        ws2['A1'] = "Material Name"
        ws2['A1'].font = header_font
        ws2['A1'].fill = header_fill
        
        for idx, material in enumerate(data.get('materials', []), start=2):
            ws2[f'A{idx}'] = material
        
        # Optimization History Sheet
        ws3 = wb.create_sheet("Optimization History")
        ws3['A1'] = "Iteration"
        ws3['B1'] = "Energy Value"
        ws3['A1'].font = header_font
        ws3['A1'].fill = header_fill
        ws3['B1'].font = header_font
        ws3['B1'].fill = header_fill
        
        history = data.get('history', [])
        for idx, energy_val in enumerate(history, start=2):
            ws3[f'A{idx}'] = idx - 1
            ws3[f'B{idx}'] = round(energy_val, 6)
        
        # Adjust column widths
        for ws in [ws1, ws2, ws3]:
            for column in ws.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(cell.value)
                    except:
                        pass
                adjusted_width = (max_length + 2)
                ws.column_dimensions[column[0].column_letter].width = adjusted_width
        
        # Save to BytesIO
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'quantum_optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Quantum Photocatalyst Optimizer (Enhanced with Excel Export)...")
    app.run(debug=True, host="0.0.0.0", port=5000)
