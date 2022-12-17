from pytorch_lightning.loggers import WandbLogger
import wandb
import pytorch_lightning as pl

class LogPredictionSamplesCallback(pl.Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from the first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' 
                for y_i, y_pred in zip(y[:n], outputs[:n])]
            

            # Option 2: log images and predictions as a W&B Table
            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred]] 
            for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n])):
                wandb_logger.log_table(
                    key='image_table',
                    columns=columns,
                    data=data)   
            
# callbacks
class PRMetrics(pl.Callback):
    """ Custom callback to compute per-class PR & ROC curves
    at the end of each training epoch
    """
    def __init__(self,  val_samples, num_samples=32, class_names={'Non-responder':0, 'Responder':1}):    #generator=None, num_log_batches=1):
        # self.generator = generator
        # self.num_batches = num_log_batches
        # # store full names of classes
        # self.class_names = { v: k for k, v in generator.class_indices.items() }
        # self.flat_class_names = [k for k, v in generator.class_indices.items()]

        super().__init__()
        self.num_samples = num_samples
        self.class_names = class_names
        self.val_imgs, self.val_labels = val_samples

    def on_epoch_end(self, trainer, pl_module, logs={}):
        # # collect validation data and ground truth labels from generator
        # val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
        # val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

        # # use the trained model to generate predictions for the given number
        # # of validation data batches (num_batches)
        # val_predictions = self.model.predict(val_data)
        # ground_truth_class_ids = val_labels.argmax(axis=1)
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        preds = torch.argmax(pl_module(val_imgs), -1)
        # Log precision-recall curve the key "pr_curve" is the id of the plot--do not change this if you want subsequent runs to show up on the same plot
        wandb.log({"roc_curve" : wandb.plot.roc_curve(val_labels, preds, labels=self.class_names)})

class ImagePredictionLogger(pl.Callback):
    """ callback"""
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
        })
        