- `PassID`: Unique pass identifier
- `timestamp`: Pass timestamp
- `start_pos`: 
- `end_pos`:
- `ToolIdx`: Unique identifier for the tools used (increment each times that the tools are changed)
- `PassNumber`: Number of pass per tools
- `Axe_X_master/ActualVelocity`: Actual feedrate given in mm/s
- `Broche/ActualSpeed`: Actual spindle rotation speed in tr/min
- `Broche/StatusTorqueData.ActualTorque`: Spindle torque in Newton
- `DB_PASSES/SELECTION_ALLIAGE`: Alloy selection.
    - 1: 5461. 5301
    - 2: 7196
    - 3: 6218, 638, 6910
    - 4: 6090, 6080
    - 5: 7212, 7213, 7214, 7221
    - 6: 2100
    - 7: 6212, 632
    - 99: Mix
    - 101: Mix
- `DB_PASSES/PASSE_ACTIVE.EPAISSEUR`: Plate thickness at the end of the pass
- `DB_PASSES/EPAISSEUR_BRUTE`: Initial plate thickness
- `DB_PASSES/NUMERO_OF`: Order Fabrication number
- `DB_PASSES/NUMERO_PASSE`: Number of passes per plate
- `removedMaterial`: Total material removed 
- `plate_id`: Unique identifier for each plate
- `cutting_depth`: Cutting depth
- `pass_type`: Type of pass (Blanking // Roughing // Pre-Finishing // Finishing)
- `sensor_file`: Path of the captured sensor data during the pass

# Sensor 
- AccX
- AccY
- AccZ
- Sound



# Made by Borgeat Rémy