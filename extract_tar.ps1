$rootDirectory = "D:\Machine_Learning\MinGPT\extracted_openwebtext"  # Replace with the path to your 'extracted_open_webtext' folder
$7zipPath = "C:\Program Files\7-Zip\7z.exe"  # Adjust path if 7-Zip is installed elsewhere

# Get all directories within the root directory
$directories = Get-ChildItem -Path $rootDirectory -Directory

foreach ($dir in $directories) {
    # Construct the path to the file that has the same name as the directory
    $filePath = Join-Path -Path $dir.FullName -ChildPath $dir.Name

    # Check if the file exists
    if (Test-Path $filePath) {
        # Extract the file using 7-Zip
        & $7zipPath x $filePath -o"$dir.FullName"

        # Uncomment below line if you want to remove the archive file after extraction
        Remove-Item $filePath
    } else {
        Write-Host "File $filePath not found."
    }
}
