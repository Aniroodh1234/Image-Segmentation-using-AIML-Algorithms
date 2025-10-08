def get_dataloaders(batch_size=8, num_workers=4, train_split=0.8):
    """
    Create train and validation dataloaders
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        train_split: Fraction of data to use for training (rest for validation)
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    from sklearn.model_selection import train_test_split
    import shutil
    
    # Check if processed data exists
    if not PROCESSED_DATA_DIR.exists() or len(list(PROCESSED_DATA_DIR.glob('*.jpg'))) == 0:
        print("⚠️ No processed images found!")
        print("Please run: python scripts/prepare_dataset.py")
        print("Creating dummy dataset for demonstration...")
        _create_dummy_dataset()
    
    # Get all image files from processed directory
    image_files = sorted(list(PROCESSED_DATA_DIR.glob('*.jpg')) + 
                        list(PROCESSED_DATA_DIR.glob('*.png')))
    
    if len(image_files) == 0:
        raise ValueError("No images found! Please run prepare_dataset.py first.")
    
    print(f"Found {len(image_files)} processed images")
    
    # Check if masks exist
    mask_files = sorted(list(MASKS_DIR.glob('*.png')))
    if len(mask_files) == 0:
        print("⚠️ No masks found!")
        print("Please run: python scripts/prepare_dataset.py")
        raise ValueError("Masks not found! Run prepare_dataset.py to create them.")
    
    print(f"Found {len(mask_files)} masks")
    
    # Split into train and validation
    train_files, val_files = train_test_split(
        image_files, train_size=train_split, random_state=42
    )
    
    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")
    
    # Copy files to train/val directories
    train_dir = PROCESSED_DATA_DIR / 'train'
    val_dir = PROCESSED_DATA_DIR / 'val'
    train_mask_dir = MASKS_DIR / 'train'
    val_mask_dir = MASKS_DIR / 'val'
    
    for directory in [train_dir, val_dir, train_mask_dir, val_mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Copy training files
    for f in train_files:
        img_dest = train_dir / f.name
        mask_src = MASKS_DIR / f"{f.stem}.png"
        mask_dest = train_mask_dir / f"{f.stem}.png"
        
        if not img_dest.exists():
            shutil.copy(f, img_dest)
        if mask_src.exists() and not mask_dest.exists():
            shutil.copy(mask_src, mask_dest)
    
    # Copy validation files
    for f in val_files:
        img_dest = val_dir / f.name
        mask_src = MASKS_DIR / f"{f.stem}.png"
        mask_dest = val_mask_dir / f"{f.stem}.png"
        
        if not img_dest.exists():
            shutil.copy(f, img_dest)
        if mask_src.exists() and not mask_dest.exists():
            shutil.copy(mask_src, mask_dest)
    
    # Create datasets with PRE-COMPUTED masks
    train_dataset = SegmentationDataset(
        train_dir,
        mask_dir=train_mask_dir,  # FIXED: Use pre-computed masks
        transform=get_training_augmentation(),
        create_masks=False  # FIXED: Don't create masks on-the-fly!
    )
    
    val_dataset = SegmentationDataset(
        val_dir,
        mask_dir=val_mask_dir,  # FIXED: Use pre-computed masks
        transform=get_validation_augmentation(),
        create_masks=False  # FIXED: Don't create masks on-the-fly!
    )
    
    print(f"✅ Created training dataset with {len(train_dataset)} samples")
    print(f"✅ Created validation dataset with {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader