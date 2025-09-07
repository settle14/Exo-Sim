# Mesh Files Download Instructions

## Overview
The mesh files required for this project are not included in this repository due to their large size. They can be downloaded from the original Kinesis project by amathislab.

## Download Source
**Original Repository**: [amathislab/Kinesis](https://github.com/amathislab/Kinesis)

## Required Mesh Files
The following mesh files are needed for proper functioning of this project:

### Essential Mesh Files:
- `data/xml/myotorsorigid_assets.xml` - Torso and upper body mesh assets
- `data/xml/myolegs_assets.xml` - Leg muscle and bone mesh assets
- Associated `.stl` or `.obj` mesh files referenced in these XML files

## Download Instructions

### Method 1: Clone the Original Repository
```bash
# Clone the original Kinesis repository
git clone https://github.com/amathislab/Kinesis.git

# Copy mesh files to your project
cp -r Kinesis/data/xml/*assets.xml /path/to/your/project/data/xml/
cp -r Kinesis/data/meshes/ /path/to/your/project/data/meshes/
```

### Method 2: Direct Download
1. Visit: https://github.com/amathislab/Kinesis
2. Download the repository as ZIP
3. Extract the following directories to your project:
   - `data/xml/` (containing asset XML files)
   - `data/meshes/` (containing 3D mesh files)

### Method 3: Git Sparse Checkout (Recommended for Large Repositories)
```bash
# Initialize a new git repository
git init kinesis_meshes
cd kinesis_meshes

# Add the remote repository
git remote add origin https://github.com/amathislab/Kinesis.git

# Enable sparse checkout
git config core.sparseCheckout true

# Specify which directories to download
echo "data/xml/*assets.xml" >> .git/info/sparse-checkout
echo "data/meshes/" >> .git/info/sparse-checkout

# Pull only the specified files
git pull origin main

# Copy to your project
cp -r data/ /path/to/your/project/
```

## File Structure
After downloading, your project structure should include:
```
your_project/
├── data/
│   ├── xml/
│   │   ├── myotorsorigid_assets.xml
│   │   ├── myolegs_assets.xml
│   │   └── other_asset_files.xml
│   ├── meshes/
│   │   ├── torso/
│   │   ├── legs/
│   │   └── other_mesh_directories/
│   └── other_data_files...
```

## Verification
To verify that the mesh files are correctly installed:

1. Check that the XML asset files exist:
   ```bash
   ls data/xml/*assets.xml
   ```

2. Verify mesh directories are present:
   ```bash
   ls data/meshes/
   ```

3. Run a simple test to ensure MuJoCo can load the model:
   ```bash
   python -c "import mujoco; mujoco.MjModel.from_xml_path('data/xml/myolegs_exo.xml')"
   ```

## Troubleshooting

### Common Issues:
1. **Missing XML files**: Ensure both `myotorsorigid_assets.xml` and `myolegs_assets.xml` are in `data/xml/`
2. **Missing mesh files**: Check that all `.stl` or `.obj` files referenced in XML are in `data/meshes/`
3. **Path issues**: Verify that relative paths in XML files match your directory structure

### File Size Information:
- Total mesh data: ~500MB - 1GB
- Individual mesh files: 1-50MB each
- Asset XML files: <1MB each

## Credits
- **Original Kinesis Project**: [amathislab/Kinesis](https://github.com/amathislab/Kinesis)
- **Developed by**: Mathis Lab at École Polytechnique Fédérale de Lausanne (EPFL)
- **Paper**: "Learning robust, real-time, reactive robotic movement" (Nature, 2024)

## License
Please refer to the original Kinesis repository for licensing information regarding the mesh files and assets.

## Support
If you encounter issues with downloading or using the mesh files:
1. Check the original [Kinesis repository](https://github.com/amathislab/Kinesis) for updates
2. Review the original project's documentation
3. Ensure your MuJoCo installation is compatible with the mesh file formats

---

**Note**: This project is based on the Kinesis framework. Please cite the original work when using these assets in your research.
