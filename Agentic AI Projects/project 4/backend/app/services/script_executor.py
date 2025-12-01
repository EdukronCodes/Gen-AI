"""
Script Executor Service
Executes remediation scripts safely
"""
from typing import Dict, Any
import subprocess
import asyncio
import json


class ScriptExecutor:
    """Executes remediation scripts"""
    
    async def execute(self, script: str, ticket_id: int, 
                     script_type: str = "bash") -> Dict[str, Any]:
        """
        Execute remediation script
        
        Supported types: bash, powershell, python, kubectl, terraform
        """
        try:
            if script_type == "bash":
                result = await self._execute_bash(script)
            elif script_type == "powershell":
                result = await self._execute_powershell(script)
            elif script_type == "python":
                result = await self._execute_python(script)
            elif script_type == "kubectl":
                result = await self._execute_kubectl(script)
            elif script_type == "terraform":
                result = await self._execute_terraform(script)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported script type: {script_type}"
                }
            
            return {
                "success": result.get("returncode") == 0,
                "output": result.get("stdout", ""),
                "error": result.get("stderr", ""),
                "returncode": result.get("returncode")
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_bash(self, script: str) -> Dict[str, Any]:
        """Execute bash script"""
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode()
        }
    
    async def _execute_powershell(self, script: str) -> Dict[str, Any]:
        """Execute PowerShell script"""
        process = await asyncio.create_subprocess_exec(
            "powershell", "-Command", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode()
        }
    
    async def _execute_python(self, script: str) -> Dict[str, Any]:
        """Execute Python script"""
        process = await asyncio.create_subprocess_exec(
            "python", "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode()
        }
    
    async def _execute_kubectl(self, script: str) -> Dict[str, Any]:
        """Execute kubectl commands"""
        # This would execute kubectl commands
        # In production, this should have proper authentication and safety checks
        process = await asyncio.create_subprocess_exec(
            "kubectl", *script.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode()
        }
    
    async def _execute_terraform(self, script: str) -> Dict[str, Any]:
        """Execute Terraform commands"""
        # This would execute terraform commands
        # In production, this should have proper state management
        process = await asyncio.create_subprocess_exec(
            "terraform", *script.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return {
            "returncode": process.returncode,
            "stdout": stdout.decode(),
            "stderr": stderr.decode()
        }


