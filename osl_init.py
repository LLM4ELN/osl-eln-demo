""" Expose an osl-client"""
from dotenv import load_dotenv
from os import environ
from osw.express import OswExpress, CredentialManager

load_dotenv()

cred_mngr = CredentialManager()
cred_mngr.add_credential(CredentialManager.UserPwdCredential(
    iri="llm4eln.semos.dev",
    username=environ.get("OSW_USER"),
    password=environ.get("OSW_PASSWORD"),
))

osl_client = OswExpress(domain="llm4eln.semos.dev", cred_mngr=cred_mngr)
