### Using Gated Hugging Face Models in Your Submission 🔒

The goal is to ensure your models remain secure and private while still being accessible to the evaluation servers. You can achieve this by wrapping your models as publicly visible but securely gated Hugging Face repositories.

To securely use such gated public Hugging Face models in your submissions, you must grant the `aicrowd` account access to your publicly visible but gated repository. **All repository names must include "aicrowd"** to ensure validation success.

- ✅ **Valid Example**: `team-aicrowd-my-model`
- ❌ **Invalid Example**: `team-my-model`

---

### Recommended Setup for Teams

- **Single-person teams**: Create a public, gated model under your personal Hugging Face account.
- **Multi-person teams**: Create a Hugging Face organization and manage your public gated models within this organization for better team collaboration and coordination.

**Note**: Public gated models ensure that your model is secure. Only explicitly invited accounts (such as `aicrowd`) will have access, ensuring other participants cannot view or access your submissions.

---

### Step-by-Step Guide: Creating a Public Gated Hugging Face Model

1. Log in to your [Hugging Face](https://huggingface.co/) account.
2. Click on **New Model**.
3. Enter a model name (must include "aicrowd"), set visibility to **Public**, and click **Create Model**.
4. Navigate to your model's page, click the **Settings** tab.
5. Under **Access Control**, enable **"Enable Access Requests"** to gate your model.
6. Click **Save** to apply changes.

---

### Granting Access to Required Hugging Face Accounts

To ensure your gated model can be accessed during evaluation, you must explicitly grant access to the required Hugging Face accounts.

#### Required Accounts

1. **AIcrowd account**  
   - Hugging Face username: `aicrowd`  
   - https://huggingface.co/aicrowd  

2. **Krafton evaluation account**  
   - Hugging Face username: `orak-krafton-eval`  
   - https://huggingface.co/orak-krafton-eval  

#### Steps to Grant Access

1. Go to your model’s **Settings** page on Hugging Face.
2. Under **Access Control**, ensure **“Enable Access Requests”** is turned on.
3. Click **Add Access**, search for `aicrowd`, and grant access.
4. Repeat the same step to add and grant access to `orak-krafton-eval`.

> **Note:** Access is granted instantaneously. Failure to grant access to **both** accounts may result in evaluation failures.

---

### Important Reminders

- Failure to explicitly grant access to the `aicrowd` account will result in submission failures.
- Ensure your repository name always includes the keyword "aicrowd".

