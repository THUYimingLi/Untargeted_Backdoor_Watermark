import torch


class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's training mode to `test`
        by `.eval()` only during an attack process.
    """

    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device

        self._targeted = 1
        self._attack_mode = "original"
        self._return_type = "float"

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def set_attack_mode(self, mode):
        r"""
        Set the attack mode.
  
        Arguments:
            mode (str) : 'original' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
        """
        if self._attack_mode == "only_original":
            raise ValueError(
                "Changing attack mode is not supported in this attack method."
            )

        if mode == "original":
            self._attack_mode = "original"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode == "targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._transform_label = self._get_label
        elif mode == "least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(
                mode
                + " is not a valid mode. [Options : original, targeted, least_likely]"
            )

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == "float":
            self._return_type = "float"
        elif type == "int":
            self._return_type = "int"
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == "int":
                adv_images = adv_images.float() / 255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print(
                    "- Save Progress : %2.2f %% / Accuracy : %2.2f %%"
                    % ((step + 1) / total_batch * 100, acc),
                    end="\r",
                )

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print("\n- Save Complete!")

        self._switch_model()

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels

    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels

    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images * 255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def __str__(self):
        info = self.__dict__.copy()

        del_keys = ["model", "attack"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self._attack_mode
        if info["attack_mode"] == "only_original":
            info["attack_mode"] = "original"

        info["return_type"] = self._return_type

        return (
            self.attack
            + "("
            + ", ".join("{}={}".format(key, val) for key, val in info.items())
            + ")"
        )

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == "int":
            images = self._to_uint(images)

        return images
