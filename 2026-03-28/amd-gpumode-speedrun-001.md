oldzhu@DESKTOP-JJ9N1E3:~/opencode$ curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash                                                                                                                     
🍿 Installing Popcorn CLI for Hackathon (Unix/Linux/macOS)...                                                           
✅ Detected OS: linux                                                                                                   
📥 Downloading from: https://github.com/gpu-mode/popcorn-cli/releases/latest/download/popcorn-cli-linux.tar.gz          
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current                                         
                                 Dload  Upload   Total   Spent    Left  Speed                                           
  0     0    0     0    0     0      0      0 --:--:--  0:00:02 --:--:--     0                                          
  0     0    0     0    0     0      0      0 --:--:--  0:00:02 --:--:--     0                                          
100 3305k  100 3305k    0     0  43574      0  0:01:17  0:01:17 --:--:-- 44880                                          
📦 Extracting binary...                                                                                                 
✅ Binary installed to /home/oldzhu/.local/bin/popcorn-cli                                                              
✅ Created alias: popcorn -> popcorn-cli                                                                                
✅ /home/oldzhu/.local/bin already in PATH                                                                              
                                                                                                                        
🎉 Popcorn CLI installed and ready for hackathon!                                                                       
                                                                                                                        
📋 Quick Start:                                                                                                         
   1. Restart your terminal or run: source /home/oldzhu/.bashrc                                                         
   2. Register with GitHub: popcorn-cli register github                                                                 
   3. Submit your solution: popcorn-cli submit --gpu MI300 --leaderboard amd-fp8-mm --mode test <your-file>             
                                                                                                                        
🚀 Hackathon mode features:                                                                                             
   - ✅ API URL pre-configured                                                                                          
   - ✅ GitHub authentication (no Discord setup needed)                                                                 
   - ✅ All modes available: test, benchmark, leaderboard, profile                                                      
   - ✅ Clean user identification                                                                                       
                                                                                                                        
💡 Need help? Run: popcorn-cli --help                                                                                   
🔗 Example: popcorn-cli submit --gpu MI300 --leaderboard amd-fp8-mm --mode test example.py 
